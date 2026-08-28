# Plan: Vicold.HybridAI 跨平台 C++ 推理引擎

## 目标
构建一个基于 CMake 的 C++ 高速 AI 推理引擎。当前主开发平台为 Linux DTK/HIP，硬件为最多 8 张 `gfx936` 离散 GPU（每张约 64 GiB，非统一内存）。当前交付目标是在 BF16 文本权重完整常驻设备的基础上，跑通 batch=1、短上下文、greedy 解码的 Qwen3.5/Qwen3.8 27B 端到端文本生成。架构继续保留 Windows 编译兼容、CUDA 后端以及手写 HIP kernel（Tile 路线）的扩展接口。

## 关键决策
- **产品形态**：C++ 库 + 命令行 CLI。库可独立链接，CLI 用于加载 safetensor 并执行推理。
- **后端策略**：优先使用 rocBLAS/cuBLAS/CPU BLAS；通过 `Backend` / `KernelRegistry` 抽象层预留手写 kernel（Tile 路线）替换入口。
- **多设备策略**：64 层按连续区间切分到最多 8 张 GPU，当前 8 卡配置为每卡 8 层；embedding 位于首卡，final norm 和 LM head 位于末卡。只在相邻 partition 边界传输 activation。
- **内存策略**：当前正式路径只使用离散设备内存，BF16 权重完整常驻 GPU，不使用统一内存或 CPU offload。KV cache、DeltaNet state 和 workspace 随对应层驻留其目标设备。
- **平台边界**：Linux `gfx936` 是当前构建、kernel 和推理验证平台。Windows 保持公共接口、模型/IO/tokenizer 与 HIP host-only 路径可编译，本阶段不要求运行新增 HIP device kernel。
- **首个里程碑**：先建立 Tensor、DeviceManager、Allocator、Linear Op、safetensor Loader、CLI 的最小闭环。
- **视觉策略**：提供 `enable_vision` 加载选项，默认 `false`。第一阶段仅实现文本推理，不构建视觉网络，也不读取 `model.visual.*` 权重 payload；显式开启视觉时返回未实现状态，待视觉后端完成后再启用。
- **推理目标**：正式推理路径以 GPU 为主，首要后端为 ROCm/HIP。CPU 算子只用于单元测试、数值对齐和 GPU 不可用时的诊断性 fallback，不把完整 CPU 推理作为阶段性交付目标。
- **开发规范**：
  - 所有 CUDA/HIP 原生 API（如 `hipMalloc`、`cudaMalloc`、`hipStreamSynchronize`、`rocblas_xxx`、`cublas_xxx` 等）必须封装在 `src/backends/` 对应后端实现内部。
  - 业务代码（`core/`、`memory/`、`ops/`、`runtime/`、`models/`、`io/`、`cli/`）**严禁直接包含 `<cuda_runtime.h>`、`<hip/hip_runtime.h>`、`<rocblas.h>`、`<cublas_v2.h>` 等原生头文件**，只能通过 `src/backends/interface/` 中定义的抽象接口与后端交互。
  - 抽象接口需兼容 Linux 与 Windows 双平台：所有同步原语、动态库加载、文件路径、线程/进程 API 均使用跨平台封装（`std::mutex`、`std::thread`、`std::filesystem`、CMake 平台检测）。
  - 后端具体实现按平台隔离编译：Windows 上 CUDA/HIP 后端通过 CMake option 与条件编译启用；Linux 为默认目标平台。

## 当前目标模型：Qwen3.5/Qwen3.8 27B BF16

### 模型来源
- 本地目录：`/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master`
- 协议：Apache-2.0
- 文件格式：Hugging Face Transformers，safetensors
- 架构：`Qwen3_5ForConditionalGeneration`，当前只加载 `qwen3_5_text`
- 文本权重：BF16，约 50.10 GiB；视觉权重暂不加载

### 核心架构参数
| 参数 | 值 |
|------|-----|
| 类型 | Dense 因果语言模型（含视觉编码器，推理以语言模型为主） |
| 总参数量 | 27.78 B |
| 隐藏层维度 `hidden_size` | 5120 |
| 层数 `num_hidden_layers` | 64 |
| 词表大小 `vocab_size` | 248,320（已填充） |
| 隐藏层结构 | 16 × (3 × (门控 DeltaNet → FFN) → 1 × (门控注意力 → FFN)) |
| 门控 DeltaNet 线性注意力 | QK 头数 16，V 头数 48，头维度 128 |
| 门控注意力（标准 MHA/GQA） | Q 头数 24，KV 头数 4，头维度 256，RoPE 维度 64 |
| FFN 中间维度 | 17,408 |
| 上下文长度 | 原生 262,144，可扩展至 1,000,000（YaRN） |
| 权重精度 | BF16；FP8 schema 和量化表示仅作为后续扩展保留 |
| MTP | 多 Token 预测 |

### 模型对引擎的特殊要求
1. **BF16 推理支持**：权重和主要 activation 使用 BF16，GEMM/reduction 关键路径使用 FP32 accumulation；SSM state 按模型配置使用 FP32。
2. **GQA + DeltaNet 混合注意力**：每 4 层中有 3 层是 DeltaNet（线性注意力），1 层是标准 GQA 注意力，需要分别实现。
3. **门控 FFN**：存在门控机制（类似 SwiGLU）的 FFN 结构。
4. **RoPE 与 YaRN**：标准 RoPE 维度 64，长上下文需要支持 YaRN 缩放。
5. **MTP**：多 token 预测可作为可选优化，第一阶段可先实现单步预测。
6. **思考模式/指令模式切换**：模板控制，不影响引擎核心，但 tokenizer 需处理 `enable_thinking` 模板。

### 当前显存基线
- 设备：8 × `gfx936`，每张约 63.98 GiB，非统一内存。
- 文本模型权重：约 **50.10 GiB**。
- 当前切分：首末卡各约 8.04 GiB，中间卡各约 5.67 GiB。
- 剩余显存足以容纳短上下文 KV cache、DeltaNet state、workspace 和跨卡 activation buffer。
- 第一验收目标限制为 batch=1、128–512 prompt tokens、最多 16 个新 token；跑通后再逐步扩大到 4k。长上下文、量化 KV cache 和 YaRN 优化不属于当前阶段。

## 目录结构

```
Vicold.HybridAI/
├── CMakeLists.txt              # 根构建入口
├── cmake/
│   ├── Options.cmake            # 构建选项（HYBRIDAI_ENABLE_HIP/CUDA/CPU/TESTS）
│   ├── FindROCm.cmake           # ROCm/HIP 探测
│   ├── FindCUDAExt.cmake        # CUDA 探测（预留）
│   └── CPMPackages.cmake        # CPM 第三方包管理
├── src/
│   ├── core/                    # 核心抽象：Tensor、DType、Shape、Buffer、Device
│   ├── memory/                  # Allocator、MemoryPool、UnifiedMemoryPolicy
│   ├── runtime/                 # Graph、Session、ExecutionContext、PipelineStage
│   ├── backends/                # 后端抽象与具体实现
│   │   ├── interface/           # Backend、Kernel、Event、Stream 接口
│   │   ├── hip/                 # HIP 后端实现
│   │   ├── cuda/                # CUDA 后端实现（预留目录）
│   │   └── cpu/                 # CPU 后端实现
│   ├── ops/                     # 算子注册表与实现
│   │   ├── registry.h/.cc       # OpRegistry 与 KernelSelector
│   │   ├── linear.h/.cc         # Linear（GEMM）
│   │   ├── activation.h/.cc     # ReLU/SiLU/GELU
│   │   ├── norm.h/.cc           # LayerNorm/RMSNorm
│   │   ├── rope.h/.cc           # RoPE / YaRN
│   │   ├── attention.h/.cc      # Gated GQA Attention
│   │   └── delta_net.h/.cc      # Gated DeltaNet（线性注意力）
│   ├── io/
│   │   └── safetensor_loader.h/.cc   # safetensor 解析与加载
│   ├── models/
│   │   ├── qwen3_config.h/.cc   # Qwen3.8 配置解析
│   │   └── qwen3_model.h/.cc    # Qwen3.8 模型构建与前向
│   └── cli/
│       └── hybridai_cli.cc      # 命令行入口
├── include/hybridai/            # 公共 API 头文件
├── tests/                       # GTest 单元/集成测试
├── benchmarks/                  # 性能基准（后续）
├── third_party/                 # CPM 缓存或 vendor 头文件
└── doc/
    ├── plan.md                  # 本计划
    ├── architecture.md          # 架构设计文档（后续补充）
    ├── backend_notes.md         # 后端适配笔记（后续补充）
    ├── cross_platform_notes.md  # Linux/Windows 跨平台构建与封装规范
    └── qwen3_8_27b_fp8.md       # Qwen3.8-27B-FP8 模型适配说明
```

## 阶段化实施

## 当前实施状态（2026-08-27）

已完成：

- Phase 0：CMake/C++20 工程脚手架与基础依赖。
- Windows HIP/ROCm 端到端验证：Debug 构建链接成功、HIP 设备动态枚举、`hybridai_cli devices` 列出可用 GPU、`BackendComputeTest.HipFp32GemmMatchesReference` 在 `HIP_VISIBLE_DEVICES=1` 与 `ROCBLAS_TENSILE_LIBPATH` 配置下通过。
- Linux DTK/HIP 端到端构建验证：GCC 11 + Ninja 下成功发现 `/opt/dtk` 的 HIP/rocBLAS CMake 包，核心库及 demo 构建通过，`simple_infer hip` 已在 `gfx936` 上实际运行成功。
- Linux 非统一内存多卡 BF16 加载验证：已在 8 张 `gfx936`（每张约 64 GiB）上将指定 Qwen3.8/Qwen3.5 27B BF16 文本模型完整加载到设备显存。64 层按连续区间切分，每卡 8 层；embedding 位于首卡，final norm/lm_head 位于末卡。实际文本权重约 50.10 GiB，首末卡各约 8.04 GiB，其余卡各约 5.67 GiB。
- Phase 1：Device、DType、Shape、Tensor、Backend 等核心抽象。
- Phase 2：MemoryPool、MemoryPlanner 与 CPU/HIP stub 后端。
- Phase 3：Backend/KernelRegistry 抽象及 CUDA/HIP API 封装边界。
- Phase 4：Linear、Elementwise、RMSNorm、RoPE、Softmax 等基础算子。
- Phase 5：Gated GQA Attention 与 Gated DeltaNet reference 实现，用于验证后续 GPU kernel。
- Phase 6 初版：SafeTensor 单文件读取、FP8 E4M3 基础反量化、Qwen3 配置解析。
- 真实模型 metadata 校验：已验证 `E:/models/Qwen3.8-27B-FP8` 的配置、64 个 layer 分片及关键 tensor header。
- 分片基础支持：已支持 `model.safetensors.index.json`、跨分片 tensor 查询和按需读取。
- 扩展性基础：已加入通用 `ModelWeightSchema`、Qwen 专用 schema 和 `WeightPlacementPlanner`。
- 视觉开关：已实现 `enable_vision=false` 默认策略；文本模式不会加载视觉权重 payload。
- HIP 后端基础实现：已完成 HIP allocator、buffer、stream、event、H2D/D2H/D2D copy、设备可用性检查和 rocBLAS FP32 GEMM 封装。
- HIP/rocBLAS CMake 探测：已验证 Windows ROCm SDK 的 `hip` 与 `rocblas` config package 可以被识别。
- **Windows HIP/ROCm 构建链路打通**：将 `src/backends/hip/hip_backend.cc` 改为由 MSVC 直接编译（ROCm headers 在该文件上是 host-only），彻底消除了 ROCm Clang 与 MSVC Debug CRT 的 runtime library mismatch（LNK2038）。`hybridai_test.exe` Debug 配置完整链接成功。
- 自动化测试：普通 CPU/stub 构建的 GTest 全部通过，共 62 项测试；HIP 后端注册/创建/空 kernel 注册测试通过。

进行中：

- typed GEMM 与 HIP BF16 计算路径：扩展 `Backend::gemm()` 的 dtype 合约，并使用 rocBLAS BF16 输入、FP32 accumulation。
- BF16 基础 GPU kernel：embedding、RMSNorm、residual、SwiGLU、partial RoPE、causal softmax 和 argmax。
- Qwen3.5 Full Attention、DeltaNet state/cache 和八卡短上下文前向。
- Hugging Face 逐层数值对齐，以及 reference 路径到 GPU kernel 的迁移。

尚未完成：

- 持久化 KV cache、DeltaNet recurrent state 和增量 decode。
- partition 边界 peer copy/pinned-host fallback 和流水线执行。
- Linux 完整测试依赖/CTest 验证和 CUDA 后端实现。
- 单层 Transformers 数值对齐、性能测试和最终 CLI 端到端推理验收。

### 最新构建与 GPU 验证结论

- **Linux HIP 当前实测**：在 `HIP_VISIBLE_DEVICES=0` 下，`build-linux-hip/bin/qwen_infer` 已成功加载约 50.10 GiB 的 BF16 文本权重，并完成 64 层混合 Attention/DeltaNet、MLP、final norm、lm_head 和多步 full-recompute greedy decode。推理路径当前是 reference 实现，性能尚未优化。
- **Qwen3.5 语义修复**：主 RMSNorm 使用官方的 `(1 + weight)` 参数化；DeltaNet 的 `RMSNormGated` 保持普通乘权重并在归一化后施加 `SiLU(z)`；标准 Attention 使用 q/k per-head norm、half-split RoPE、sigmoid gate，且 q/gate 按 `[head, 2*head_dim]` 交错布局拆分。
- **Tokenizer 模板**：C++ tokenizer 已支持 `<|im_end|>`、assistant generation prompt 和 `enable_thinking=false` 的空 thinking 段。当前 demo 使用关闭 thinking 的对话模板，以便先验证正常文本输出。
- **当前数值状态**：模型可以稳定运行，但尚未达到 Hugging Face 参考输出；后续仍需做逐层/逐 token 数值对齐，重点是 BF16 staging、DeltaNet conv/state、attention 和 MLP 的算子精度。
- **多卡 activation 传输**：`Tensor::to()` 的 HIP GPU-to-GPU 路径已在 backend 中区分跨设备拷贝，并使用 `hipMemcpyPeerAsync`，避免把源卡指针直接交给目标卡的普通 `hipMemcpyDeviceToDevice`。backend 的内存、拷贝、同步和 GEMM 操作都会恢复所属 HIP device；`Tensor::to()` 会在异步跨卡拷贝后同步目标设备，确保源 tensor 生命周期安全。
- **多卡稳定性验证**：已在 `HIP_VISIBLE_DEVICES=0,1` 下完成 2 卡真实 BF16 加载、64 层 full-recompute 和 8 步 greedy decode；已在 `HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7` 下完成 8 卡同样验证，进程退出码均为 0，未再出现 VMFault。此前双卡 VMFault 的直接原因是多个 HIP backend 交替使用后当前 device 上下文漂移。
- **最新 Linux HIP 验证**：`Qwen3Config` 已动态解析 `rms_norm_eps`、DeltaNet 的 key/value head 数、head dim 和卷积核尺寸；C++ tokenizer 对 `Hello, how are you?` 生成的 ID 已与 HF 基准完全一致：`[248045, 846, 198, 9419, 11, 1204, 513, 488, 30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]`。单卡 `HIP_VISIBLE_DEVICES=0` 下仍可完成 64 层 full-recompute 推理，未出现 NaN/Inf，但 greedy 输出仍主要为 `271`/`198`（空白/换行），因此模型数值尚未对齐 HF；已加入 compact layer statistics 作为后续定位基础。

- `build/`：CPU/stub 配置稳定构建通过，62 项测试通过。
- `build-debug/`（`-D HYBRIDAI_ENABLE_HIP=ON`）：完整 Debug 构建通过，`hybridai.lib` 与 `hybridai_test.exe` 链接成功，无 LNK2038 runtime mismatch。
- Windows HIP 编译方案：放弃 ROCm Clang custom command，改用 MSVC 直接编译 `hip_backend.cc`。ROCm include 路径与 `__HIP_PLATFORM_AMD__` 宏通过 CMake `set_source_files_properties` 仅作用于该源文件，主库仍正常链接 `hip::host` 与 `roc::rocblas` 导入库。
- HIP 测试程序：在配置 `ROCBLAS_TENSILE_LIBPATH` 指向 `.../rocblas/library/gfx1150` 并设置 `HIP_VISIBLE_DEVICES=1` 后，`BackendComputeTest.HipFp32GemmMatchesReference` 实测通过（约 330 ms）。`hybridai_cli devices` 正确列出可见的 HIP 设备。
- 结论：Windows 下 HIP/ROCm 从 CMake 配置、编译、链接到程序启动，再到真实 GPU GEMM 执行的完整链路已打通。当前环境需通过 `HIP_VISIBLE_DEVICES=1` 过滤掉不支持的 `gfx1010` 设备。

### Windows/Linux 构建策略

- `.vcxproj`、`.sln` 和其他 Visual Studio 工程文件均为 CMake 生成物，不纳入 Git；构建目录使用 `build/`、`build-debug/`、`build-hip/` 或 `build-linux/` 分离。
- Windows：继续使用 Visual Studio generator；`hip_backend.cc` 仅使用 HIP host API，因此由 MSVC 直接编译，无需 `hipcc` custom command。CMake 通过 `set_source_files_properties` 只为该文件添加 ROCm include 路径与 `__HIP_PLATFORM_AMD__` 宏。
- Linux：优先使用 Ninja 或 Unix Makefiles，并使用 CMake 原生 HIP language/`hipcc` 集成；Windows 的 MSVC 直接编译方案不适用于 Linux。
- Linux 当前 host-only HIP 后端源可由 GCC/Clang 配合 `hip::host` 编译；后续加入手写 HIP kernel 时，再为 kernel 源启用 CMake HIP language/`hipcc`。DTK 这类分散安装布局通过 `HYBRIDAI_ROCM_ROOT` 统一定位。
- 两个平台共享 `Backend`、Tensor、模型和 IO 抽象，平台差异限制在后端实现及 CMake 条件分支中。

### Windows ROCm 运行环境配置

在 Windows 上运行 HIP/rocBLAS 测试或可执行文件时，可能需要设置以下环境变量：

- `ROCBLAS_TENSILE_LIBPATH`：指向 ROCm SDK 中 rocBLAS Tensile kernel 数据目录，例如：
  ```powershell
  $env:ROCBLAS_TENSILE_LIBPATH = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'
  ```
  该目录下需要存在 `TensileLibrary.dat` 主文件（当前 SDK 提供的是 `TensileLibrary_lazy_gfx1150.dat`，可复制一份重命名为 `TensileLibrary.dat`）。
- `HIP_VISIBLE_DEVICES`：当机器存在多个 AMD GPU 且只有部分受 ROCm 支持时，用该变量过滤可见设备。例如当前机器 device 0 为 `gfx1010`、device 1 为 `gfx1150`，只有后者能运行 SDK 中的 rocBLAS kernel，因此需要：
  ```powershell
  $env:HIP_VISIBLE_DEVICES = '1'
  ```
- `PATH`：确保 `amdhip64.dll`、`rocblas.dll` 等 ROCm 运行时 DLL 可被加载。通常 Python ROCm 包安装目录下的 `bin` 已加入 PATH，否则需手动添加。

## 架构扩展原则

本项目不能假设所有模型或所有设备都采用相同的权重布局。后续实现必须遵守以下边界：

1. **模型无关的文件层**：`SafeTensorLoader` 只负责单文件/目录/index、header 解析和按需字节读取，不包含 Qwen 网络结构判断。
2. **模型专用的 schema 层**：Qwen 的 tensor 命名、层结构、权重别名和 scale 配对放在 `Qwen3WeightSchema`/`Qwen3WeightLoader` 中。未来通过新增 `LlamaWeightSchema`、`DeepSeekWeightSchema` 等支持其他模型，不修改通用加载器。
3. **模型无关的量化层**：FP8、INT8、INT4 等采用统一量化权重描述，scale 形式通过 metadata 描述，不能把二维 block scale 写死到 Qwen 前向代码。
4. **模型无关的放置层**：权重放置由 `WeightPlacementPlanner` 决定，模型代码只提供 tensor 名称、层编号、大小和生命周期信息。
5. **单文件与分片并存**：同一模型可以是一个 `model.safetensors`，也可以由 index 指向多个分片；业务层不能依赖具体文件名或分片数量。
6. **切片策略可配置**：默认支持不切片/单设备加载；资源不足时才启用按层切片、CPU offload 或多设备放置。切片不是模型格式的强制要求。
7. **设备策略可扩展**：设备列表、容量、统一内存能力、带宽和优先级通过运行时配置提供，不能在 Qwen 类中写死 iGPU/dGPU 数量。
8. **后端隔离**：模型、算子编排和 IO 层不得包含 CUDA/HIP/rocBLAS/cuBLAS 原生头文件或调用，只能访问 `Backend` 抽象。
9. **多模态按需加载**：视觉等可选模态必须由模型加载选项显式开启。默认文本模式不得加载视觉 tensor payload，避免无效内存和磁盘带宽占用。

## 可配置权重放置方案

统一使用 `WeightShardingConfig` 描述部署策略：

| 模式 | 行为 | 适用场景 |
|------|------|----------|
| `Replicated` | 权重复制到配置的设备 | 单设备、统一内存或显存足够 |
| `Layer` | 按层分配到设备 | 多设备协同、显存不足 |
| `ManualRanges` | 用户指定层范围与目标设备 | 固定部署拓扑 |
| `MemoryBalanced` | 根据设备可用容量自动平衡 | 异构 iGPU/dGPU/CPU |
| `CPUOffload` | 权重常驻 CPU，当前层按需搬运 | 单卡显存严重不足 |

配置至少应包含：

- 设备列表及设备类型；
- 每个设备的可用内存预算；
- 是否允许统一内存；
- 是否允许复制公共权重；
- 层切分范围或自动平衡策略；
- 当前层预取数量；
- 内存不足时的 fallback 顺序。

示例部署语义：

```text
单设备模式：        全部权重 -> CPU 或一个 GPU
按层模式：          layer[0..N] -> iGPU，layer[N+1..63] -> dGPU
CPU offload 模式：   CPU 保存全部权重，GPU 只保留当前计算层
混合模式：          embedding/final norm -> CPU，其余层按容量分配
```

## 模型插件边界

每个模型适配器应至少提供以下组件：

```text
ModelConfig              # 配置解析
ModelWeightSchema         # tensor name -> canonical key/layer/scale
ModelWeightLoader         # 组装模型权重对象
ModelArchitecture         # 层结构和前向编排
ModelTokenizerAdapter     # tokenizer 与 chat template
```

Qwen3.8 只是第一个适配器，不应成为核心运行时的硬编码分支。

### Phase 0: 脚手架与构建系统
1. 创建根 `CMakeLists.txt`，C++20 标准，CMake 3.20+。
2. 在 `cmake/Options.cmake` 中定义构建选项：
   - `HYBRIDAI_ENABLE_HIP` / `HYBRIDAI_ENABLE_CUDA` / `HYBRIDAI_ENABLE_CPU`
   - `HYBRIDAI_BUILD_TESTS` / `HYBRIDAI_BUILD_CLI`
3. 配置 CPM 包管理器，引入：
   - `fmt`、`spdlog`、`nlohmann_json`
   - `GoogleTest`
   - `safetensors-cpp`（或自实现 header-only 解析器）
   - `sentpiece` / `llama.cpp` tokenizer（用于 Qwen3.8 tokenizer）
4. 建立 `src/`、`include/`、`tests/`、`doc/` 目录结构。
5. **验证**：
   ```bash
   cmake -B build -D HYBRIDAI_ENABLE_CPU=ON -D HYBRIDAI_BUILD_TESTS=ON
   cmake --build build
   ```

### Phase 1: 核心抽象
1. 实现 `hybridai::Device`：封装 device id、type（CPU/iGPU/dGPU/unknown）、backend 名称、统一内存支持标志。
2. 实现 `hybridai::DType`：支持 FP32/FP16/BF16/FP8_E4M3/INT8/INT32 等枚举与字节大小查询。
3. 实现 `hybridai::Tensor`：Shape + Stride + DType + Buffer + Device，支持 `to(device)` 与 `contiguous()`。
4. 实现 `hybridai::Stream` 与 `hybridai::Event`：跨后端抽象，HIP 下对应 `hipStream_t`/`hipEvent_t`。
5. 实现 `Platform` 工具层：封装 Windows/Linux 下的动态库加载、线程亲和性、路径处理、编译器宏差异。
6. **验证**：GTest 测试 Tensor 构造、跨设备拷贝、事件同步；CI/本地分别在 Windows 与 Linux 下编译通过。

### Phase 2: 内存管理
1. 实现 `Allocator` 接口：`allocate`、`deallocate`、`memory_type`。
2. 实现 `DeviceAllocator`：dGPU 用 `hipMalloc`/`cudaMalloc`。
3. 实现 `UnifiedAllocator`：iGPU 用 `hipMallocManaged`/`cudaMallocManaged`。
4. 实现 `MemoryPool`：基于 bucket 的缓存分配器，减少 `hipMalloc` 开销。
5. 实现 `MemoryPlanner`：根据模型大小、设备显存、是否统一内存，选择 Allocator 类型与 offload 策略。
6. **验证**：分配/释放不同大小内存，池命中统计，MemoryPlanner 在 iGPU/dGPU 上选择正确类型。

### Phase 3: 后端抽象与 HIP 实现
1. 定义 `Backend` 接口：`name()`、`create_stream()`、`create_allocator()`、`memcpy_h2d/d2h/d2d()`、`synchronize()`、`gemm()`、`register_kernel()`。
2. 实现 `HipBackend`：封装 rocBLAS 句柄、hipStream、设备属性探测。
3. 实现 `CpuBackend`：使用 OpenBLAS/MKL/oneDNN 作为 fallback；仅依赖 `Backend` 接口，不直接调用 CUDA/HIP API。
4. 设计 `KernelRegistry`：允许手写 kernel 按（op, device, dtype）三元组注册，运行时可替换默认 BLAS 实现。
5. **验证**：Backend 工厂按设备类型创建正确后端；手写空 kernel 注册/反注册测试；`grep -R "hipMalloc\|cudaMalloc" src/core src/ops src/runtime src/models src/io src/cli` 无结果。

当前状态：CPU/stub 路径已完成；Windows 下 HIP/rocBLAS 代码已由 MSVC 直接编译、主库与测试可执行文件完整链接成功。设置 `HIP_VISIBLE_DEVICES=1` 后，真实 GPU（gfx1150）上的 rocBLAS FP32 GEMM 测试通过。

### Phase 4: 基础算子与注册表
1. 实现 `OpRegistry` 与 `KernelSelector`：根据 Tensor 设备、DType、全局策略选择 kernel。
2. 实现 `Linear` 算子：调用后端 `gemm()`，优先 rocBLAS/cuBLAS，预留手写 tile kernel 注册入口。
3. 实现 `Activation`（SiLU/GELU/Swish）、`RMSNorm`、`RoPE` 等基础算子。
4. 实现 `Tensor` 的链式执行包装。
5. **验证**：Linear 前向结果与 PyTorch/NumPy 对齐；GTest 对比 FP32 精度。

### Phase 5: Qwen3.8 注意力与 DeltaNet 算子
1. 实现 `GatedGQAAttention`：
   - Q 投影 5120→6144（24 头 × 256）
   - K/V 投影 5120→2048（4 KV 头 × 256）
   - RoPE 应用于 Q/K 的 64 维 rope 部分
   - 标准 Softmax × V
2. 实现 `GatedDeltaNet`：
   - 线性注意力：QK 头 16、V 头 48、头维度 128
   - 门控状态空间形式（保留 decay/gate 参数）
   - 可先按 recurrent 形式实现，再优化为并行形式
3. **验证**：与 Hugging Face Transformers 输出对齐（单层，短序列）。

### Phase 6: safetensor 加载与 FP8 反量化
1. 实现 `SafeTensorLoader`：读取 `.safetensors` header JSON，解析 tensor metadata，按名字映射加载到 `Tensor`。
2. 实现目录/index-aware SafeTensor loader：同时支持单文件、未切片目录和多个分片；读取 index 后按 tensor 按需打开 shard。
3. 实现通用 `QuantizedWeight` 与 scale metadata：支持 per-tensor、per-row、per-column 和二维 block scale。
4. 实现 `FP8Dequantizer`：将 FP8 E4M3 块量化权重按 128 block 反量化为 BF16/FP16，保留 FP8 GEMM 的后端扩展入口。
5. 实现通用 `ModelWeightSchema`，再实现 Qwen3 专用 schema 和 scale 配对规则。
6. 实现 `Qwen3Config`：解析 `config.json` 中的模型超参数。
7. **验证**：只读 header 扫描全部真实分片，检查权重名、形状、dtype、offset；对单个 Linear 权重反量化后与 BF16 基模型对比误差。

当前状态：SafeTensor 单文件/分片读取、真实模型 metadata 检查、FP8 E4M3 基础表示和 Qwen schema 已完成；GPU-safe 转换仍待实现。

### Phase 7: Qwen3.8 模型构建与前向
1. 实现通用 `ModelBuilder` 生命周期和设备放置接口。
2. 实现 `Qwen3ModelBuilder`：根据 `config.json`、Qwen schema 与 safetensor 权重名构建完整的 64 层网络。
3. 实现按层懒加载/释放，避免一次性将约 30.89GB 权重反量化到单一设备。
4. 实现 `Qwen3Model::Forward(input_ids)`：
   - Embedding lookup
   - 64 层循环（DeltaNet × 3 + GQA Attention × 1 为一组，共 16 组）
   - RMSNorm + LM Head
   - Sampling（greedy / top-k / top-p）
5. 支持短上下文（4k / 8k）先跑通。
6. **验证**：GPU 输出与 reference/Transformers 对齐；同一权重在单设备和切分设备模式下输出一致。

当前状态：尚未进入完整模型 GPU 前向；下一阶段先实现 GPU 单算子和最小单层路径，不回退为完整 CPU 推理交付。

### Phase 8: 多设备切分与流水线
1. 将 `WeightPlacementPlanner` 扩展为 `Replicated`、`Layer`、`ManualRanges`、`MemoryBalanced` 和 `CPUOffload`。
2. 实现 `PipelineStage`：持有子图、目标设备、输入输出 buffer。
2. 实现 `StageConnector`：管理相邻 stage 之间的 `d2d` 或 `d2h/h2d` 传输。
3. 实现 `PipelineExecutor`：按拓扑顺序执行 stages，使用 Event 同步。
4. 实现切分策略：
   - 按层数切分到 iGPU/dGPU
   - 由于模型 27.8GB 远大于单卡显存，必要时引入 CPU offload
5. 允许同一模型在单设备不切片模式下运行，不强制创建多设备 stage。
6. **验证**：在单设备、CPU offload 和两设备切分模式运行部分层，中间传输正确，输出与单设备一致。

### Phase 9: CLI 与可观测性
1. 实现 `hybridai_cli`：
   - `devices`：列出可用设备
   - `run --model <dir> --prompt <...>`：加载模型并生成文本
   - `bench --model <dir>`：基准测试
2. 集成 `spdlog` 日志，支持 HIP 错误码打印。
3. **验证**：CLI 能正确列出 device#0 (gfx1010) 与 device#1 (gfx1150)；加载 Qwen3.8-27B-FP8 并完成一次短 prompt 推理。

## 关键架构参考
- `Backend` 接口位于 `src/backends/interface/backend.h`，所有 GEMM、内存拷贝、同步操作均通过它派发。
- `KernelRegistry` 位于 `src/ops/registry.h`，支持运行时按 `KernelKey{op_name, device_type, dtype}` 注册自定义 kernel，为 Tile 路线替换 rocBLAS 预留入口。
- `MemoryPlanner` 位于 `src/memory/memory_planner.h`，决策函数 `SelectAllocator(Device, model_bytes)` 决定使用设备内存还是统一内存。
- `PipelineExecutor` 位于 `src/runtime/pipeline_executor.h`，通过 `StageConnector` 显式处理 iGPU/dGPU 间 PCIe 传输，避免统一内存隐式 page fault。
- `FP8Dequantizer` 位于 `src/ops/fp8_dequant.h`，负责将 Qwen3.8-27B-FP8 的块量化权重在线反量化为 BF16。
- `Qwen3Config` / `Qwen3Model` 位于 `src/models/`，封装目标模型特定逻辑。
- `Platform` 工具层位于 `src/core/platform.h`，封装 Windows/Linux 差异，确保业务代码双平台可编译。

## CUDA/HIP 封装规范
1. **唯一接触点**：只有 `src/backends/hip/`、`src/backends/cuda/`、`src/backends/interface/` 可以包含原生 GPU 头文件；其余模块通过 `Backend`、`Stream`、`Event`、`Buffer`、`Device` 等抽象访问硬件。
2. **禁止裸 API**：业务代码中禁止出现 `hipMalloc`、`cudaMalloc`、`hipMemcpy`、`cudaMemcpy`、`hipStreamSynchronize`、`cudaStreamSynchronize`、`rocblas_xxx`、`cublasXxx` 等符号。
3. **后端工厂**：`BackendFactory` 在运行时/编译期根据 CMake option 决定加载哪个后端；业务代码不感知当前是 HIP、CUDA 还是 CPU。
4. **错误码统一**：所有后端错误统一转换为 `hybridai::Status` / `hybridai::Error`，上层按统一错误码处理。
5. **跨平台构建**：CMake 中通过 `CMAKE_SYSTEM_NAME` 区分 Windows/Linux，平台相关依赖（如 `dlopen` vs `LoadLibrary`）封装在 `Platform` 层。
6. **验证命令**：
   ```bash
   # 检查业务代码是否违规调用 CUDA/HIP 原生 API
   grep -R "hipMalloc\|cudaMalloc\|hipMemcpy\|cudaMemcpy\|hipStream\|cudaStream\|rocblas_\|cublas" \
         src/core src/memory src/ops src/runtime src/models src/io src/cli \
         include/hybridai
   ```
   预期输出为空。

## 风险与边界
- **BF16 正确性优先**：先实现 BF16 输入、FP32 accumulation 的可靠路径，再考虑 FP8 GEMM 和 kernel 融合。
- **DeltaNet 实现复杂**：必须以当前 Transformers 实现和导出的中间 tensor/state fixture 为事实来源，不能继续扩展现有固定 gate 占位算法。
- **多卡传输拓扑未知**：需要运行时探测 peer access；不支持时必须提供 pinned-host staging fallback。
- **显存预算不能只计算权重**：加载前还要为 KV cache、DeltaNet FP32 state、workspace、rocBLAS 临时空间和跨卡双缓冲预留容量。
- **多设备协同复杂度**：初期仅实现 layer-level 切分，不实现 expert-level 或 pipeline-parallel 高级调度。
- **视觉编码器暂时跳过**：先聚焦语言模型推理，视觉编码器后续作为扩展。
- 跨平台构建复杂度：Windows 上 ROCm/HIP 与 CMake/MSVC 的兼容性较弱，但当前已通过 MSVC 直接编译 HIP host API 的方式打通；CUDA 后端在 Windows 上更易验证。
- **封装泄漏风险**：需通过代码审查与自动化 grep 脚本确保业务代码不直接包含 CUDA/HIP 头文件。

## 验证清单
1. `cmake --build build` 在 `-D HYBRIDAI_ENABLE_HIP=ON` 下通过。
2. GTest 全部通过：Tensor、Allocator、Backend、Linear、FP8Dequantizer、SafeTensorLoader。
3. CLI `hybridai_cli devices` 正确列出当前可见设备；在多 GPU 环境中配合 `HIP_VISIBLE_DEVICES` 过滤后只列出可用设备。
4. 单层 GQA Attention 与 Transformers 输出误差 < 1e-3。
5. CLI `hybridai_cli run` 在短 prompt 上完成一次 Qwen3.8-27B-FP8 前向推理。
6. 两设备切分运行输出与单设备运行输出在 FP32 绝对误差 < 1e-4 内一致。

## 当前下一步执行顺序

1. ✅ 已在当前 AMD GPU（gfx1150，通过 `HIP_VISIBLE_DEVICES=1` 过滤）上验证 HIP/rocBLAS FP32 GEMM 实际计算成功。下一步：把环境变量配置固化为开发脚本或 CMake/CTest 环境，避免每次手动设置。
2. 完成二维 FP8 block scale 反量化和 `QuantizedWeight` 通用表示。
3. 对真实 Qwen3.8-27B-FP8 全部分片执行 metadata 扫描和 scale 配对校验。
4. 实现 `Qwen3LayerWeights`、`Qwen3ModelWeights` 和按层懒加载。
5. 实现单层 Qwen3.8 GPU 前向；CPU reference 仅用于数值对照，再扩展到 64 层和短 prompt 推理。
6. 将模型权重放置接入通用 planner，验证不切片、按层切分和 CPU offload 三种模式。
7. 实现多设备流水线、CLI 推理、性能统计和 CUDA backend。
