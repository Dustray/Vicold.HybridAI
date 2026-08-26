# Plan: Vicold.HybridAI 跨平台 C++ 推理引擎

## 目标
构建一个基于 CMake 的 C++ 高速 AI 推理引擎，初始目标平台为 ROCm/HIP，支持 iGPU（gfx1150，统一内存）+ dGPU（gfx1010，独立显存）协同推理，模型输入为 Dense safetensor 格式。架构预留未来向 CUDA、手写 HIP kernel（Tile 路线）扩展的接口。

## 关键决策
- **产品形态**：C++ 库 + 命令行 CLI。库可独立链接，CLI 用于加载 safetensor 并执行推理。
- **后端策略**：优先使用 rocBLAS/cuBLAS/CPU BLAS；通过 `Backend` / `KernelRegistry` 抽象层预留手写 kernel（Tile 路线）替换入口。
- **多设备策略**：同一模型按层/阶段切分到 iGPU 与 dGPU，PCIe 传输由 `PipelineStage` 显式管理。
- **内存策略**：运行时启发式选择——dGPU 优先使用 `hipMalloc` 设备内存；iGPU 优先使用 `hipMallocManaged` 统一内存；模型大小超过显存时自动 fallback 到 CPU/分层加载。
- **平台边界**：gfx1010 仅作为 HIP 编译目标或 CPU fallback，不承诺运行成功（ROCm 官方不支持 gfx1010）。
- **首个里程碑**：先建立 Tensor、DeviceManager、Allocator、Linear Op、safetensor Loader、CLI 的最小闭环。
- **视觉策略**：提供 `enable_vision` 加载选项，默认 `false`。第一阶段仅实现文本推理，不构建视觉网络，也不读取 `model.visual.*` 权重 payload；显式开启视觉时返回未实现状态，待视觉后端完成后再启用。
- **推理目标**：正式推理路径以 GPU 为主，首要后端为 ROCm/HIP。CPU 算子只用于单元测试、数值对齐和 GPU 不可用时的诊断性 fallback，不把完整 CPU 推理作为阶段性交付目标。
- **开发规范**：
  - 所有 CUDA/HIP 原生 API（如 `hipMalloc`、`cudaMalloc`、`hipStreamSynchronize`、`rocblas_xxx`、`cublas_xxx` 等）必须封装在 `src/backends/` 对应后端实现内部。
  - 业务代码（`core/`、`memory/`、`ops/`、`runtime/`、`models/`、`io/`、`cli/`）**严禁直接包含 `<cuda_runtime.h>`、`<hip/hip_runtime.h>`、`<rocblas.h>`、`<cublas_v2.h>` 等原生头文件**，只能通过 `src/backends/interface/` 中定义的抽象接口与后端交互。
  - 抽象接口需兼容 Linux 与 Windows 双平台：所有同步原语、动态库加载、文件路径、线程/进程 API 均使用跨平台封装（`std::mutex`、`std::thread`、`std::filesystem`、CMake 平台检测）。
  - 后端具体实现按平台隔离编译：Windows 上 CUDA/HIP 后端通过 CMake option 与条件编译启用；Linux 为默认目标平台。

## 目标模型：Qwen3.8-27B-FP8

### 模型来源
- 模型库：[ModelScope - Qwen/Qwen3.8-27B-FP8](https://www.modelscope.cn/models/Qwen/Qwen3.8-27B-FP8)
- 协议：Apache-2.0
- 文件格式：Hugging Face Transformers，safetensors
- 总大小：约 30.89 GB

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
| 量化 | 细粒度 FP8 E4M3，块大小 128；部分张量为 BF16 |
| MTP | 多 Token 预测 |

### 模型对引擎的特殊要求
1. **FP8 推理支持**：权重以 FP8 E4M3 存储，需要反量化到 BF16/FP16 后再计算，或在支持的硬件上做 FP8 GEMM。
2. **GQA + DeltaNet 混合注意力**：每 4 层中有 3 层是 DeltaNet（线性注意力），1 层是标准 GQA 注意力，需要分别实现。
3. **门控 FFN**：存在门控机制（类似 SwiGLU）的 FFN 结构。
4. **RoPE 与 YaRN**：标准 RoPE 维度 64，长上下文需要支持 YaRN 缩放。
5. **MTP**：多 token 预测可作为可选优化，第一阶段可先实现单步预测。
6. **思考模式/指令模式切换**：模板控制，不影响引擎核心，但 tokenizer 需处理 `enable_thinking` 模板。

### 显存估算（FP8 权重 + BF16 KV Cache）
- 模型权重：约 27.78B × 1 byte ≈ **27.8 GB**（FP8），反量化后约 55.6 GB（BF16）
- KV Cache（原生 262k 上下文、64 层、batch=1）：约 2 × 64 × 262144 × (5120 × 1/8) × 2 byte ≈ **41.9 GB**
- 因此：
  - 单卡 7.98 GB dGPU 或 13.11 GB iGPU 无法直接放下完整模型或长上下文 KV。
  - 必须采用 **分层/分阶段多设备切分** + **上下文长度限制** + **量化 KV Cache**。
  - 实用目标：先以短上下文（如 4k / 8k / 32k）运行，验证端到端推理；长上下文作为后续优化。

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

## 当前实施状态（2026-08-26）

已完成：

- Phase 0：CMake/C++20 工程脚手架与基础依赖。
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
- Windows 构建隔离：由于当前 Visual Studio generator 不支持原生 HIP language，已改为由 `hipcc` custom command 编译 HIP 对象，再链接到 C++ 主库。
- 自动化测试：普通 CPU/stub 构建的 GTest 全部通过，共 62 项测试。

进行中：

- Windows 下 HIP custom command 的可重复构建：hipcc 手动编译已成功，但 MSBuild 执行时仍存在 MSVC `<cmath>` 与 HIP math forward declaration 冲突。
- HIP 对象与 Visual Studio Debug 目标的 ABI/CRT 对齐，已确认需要使用 `_ITERATOR_DEBUG_LEVEL=2`、`-D_DEBUG` 和 `-fms-runtime-lib=dll_dbg`。
- 真实 GPU GEMM smoke test 的运行环境整理，包括 ROCm DLL、rocBLAS Tensile 数据文件和设备选择。
- 二维 FP8 block scale 反量化与通用量化权重表示。

尚未完成：

- Qwen3.8 权重对象和完整 64 层前向。
- 可配置的完整多设备流水线与 CPU offload。
- Linux ROCm 构建验证和 CUDA 后端实现。
- 真实 HIP GPU kernel、性能测试和 CLI 端到端推理。

### 最新构建与 GPU 验证结论

- `build/`：CPU/stub 配置稳定构建通过，62 项测试通过。
- `build-hip/`：HIP/rocBLAS 配置成功；手动调用 Windows ROCm SDK 的 `hipcc` 可以编译 `hip_backend.cc`，并且使用 Debug DLL CRT 参数后可以与 Debug 主库链接。
- MSBuild custom command：仍会在 HIP 编译阶段触发 `<cmath>` 与 `__clang_cuda_math_forward_declares.h` 的 `isgreater` 等重载冲突，因此当前不能宣称 HIP CMake 构建已经完全打通。
- HIP 测试程序：在补充 ROCm DLL 路径后能够启动，但当前默认选择的 device 0（`gfx1010`）缺少 rocBLAS 所需的 `TensileLibrary.dat`，测试异常退出。需要先实现真实设备探测/选择，并确认 `gfx1150` 的 rocBLAS 数据文件与运行支持。
- 结论：GPU-first 路线已经完成后端代码和链接层的基础打通，但真实设备上的 rocBLAS GEMM 尚未验证成功；后续不能将 CPU/stub 测试结果当作 GPU 推理完成。

### Windows/Linux 构建策略

- `.vcxproj`、`.sln` 和其他 Visual Studio 工程文件均为 CMake 生成物，不纳入 Git；构建目录使用 `build/`、`build-hip/` 或 `build-linux/` 分离。
- Windows：继续使用 Visual Studio generator；HIP 源文件通过独立 `hipcc` custom command 编译，避免向 MSVC 目标传播 `hip::device` 的 `-x hip` 参数。
- Linux：优先使用 Ninja 或 Unix Makefiles，并使用 CMake 原生 HIP language/`hipcc` 集成；不复用 Windows 的 `.vcxproj` 和 custom object 方案。
- 两个平台共享 `Backend`、Tensor、模型和 IO 抽象，平台差异限制在后端实现及 CMake 条件分支中。

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

当前状态：CPU/stub 路径已完成；Windows HIP/rocBLAS 代码和主库链接已完成初步验证，但 custom hipcc 的 MSBuild 可重复构建及真实设备运行仍未闭环。

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
- **gfx1010 不支持官方 ROCm**：`HipBackend` 编译目标可包含 gfx1010，但运行测试以 CPU 或 iGPU 为主；dGPU 仅做编译兼容性验证。
- **27.8B FP8 模型远超单卡显存**：必须分层/多设备/CPU offload，初期以短上下文、部分层验证为主。
- **FP8 计算支持有限**：ROCm/rocBLAS 对 FP8 GEMM 的支持取决于 GPU 架构（MI 系列较完整，gfx1010/gfx1150 不确定），可能需要在线反量化到 BF16/FP16 计算。
- **DeltaNet 实现复杂**：线性注意力有多种实现形式（recurrent/parallel），先从简单 recurrent 开始，后续再优化。
- **iGPU 统一内存性能**：gfx1150 为集成显卡，统一内存虽可零拷贝，但带宽与延迟需实测，必要时回退到显式拷贝。
- **多设备协同复杂度**：初期仅实现 layer-level 切分，不实现 expert-level 或 pipeline-parallel 高级调度。
- **视觉编码器暂时跳过**：先聚焦语言模型推理，视觉编码器后续作为扩展。
- **跨平台构建复杂度**：Windows 上 ROCm/HIP 与 CMake/MSVC 的兼容性较弱，初期以 Linux 为主要开发平台，Windows 通过 CPU-only 模式先验证；CUDA 后端在 Windows 上更易验证。
- **封装泄漏风险**：需通过代码审查与自动化 grep 脚本确保业务代码不直接包含 CUDA/HIP 头文件。

## 验证清单
1. `cmake --build build` 在 `-D HYBRIDAI_ENABLE_HIP=ON` 下通过。
2. GTest 全部通过：Tensor、Allocator、Backend、Linear、FP8Dequantizer、SafeTensorLoader。
3. CLI `hybridai_cli devices` 正确列出 device#0 (gfx1010) 与 device#1 (gfx1150)。
4. 单层 GQA Attention 与 Transformers 输出误差 < 1e-3。
5. CLI `hybridai_cli run` 在短 prompt 上完成一次 Qwen3.8-27B-FP8 前向推理。
6. 两设备切分运行输出与单设备运行输出在 FP32 绝对误差 < 1e-4 内一致。

## 当前下一步执行顺序

1. 完成二维 FP8 block scale 反量化和 `QuantizedWeight` 通用表示。
2. 对真实 Qwen3.8-27B-FP8 全部分片执行 metadata 扫描和 scale 配对校验。
3. 实现 `Qwen3LayerWeights`、`Qwen3ModelWeights` 和按层懒加载。
4. 实现单层 Qwen3.8 GPU 前向；CPU reference 仅用于数值对照，再扩展到 64 层和短 prompt 推理。
5. 将模型权重放置接入通用 planner，验证不切片、按层切分和 CPU offload 三种模式。
6. 在不泄漏 HIP/CUDA API 的前提下实现 GPU backend 的 GEMM、基础算子和异步传输。
7. 最后实现多设备流水线、CLI 推理、性能统计和 CUDA backend。
