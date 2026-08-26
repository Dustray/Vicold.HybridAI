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
2. 实现 `FP8Dequantizer`：将 FP8 E4M3 块量化权重按 128 块大小反量化为 BF16/FP16。
3. 实现 `Qwen3Config`：解析 `config.json` 中的模型超参数。
4. **验证**：加载 Qwen3.8-27B-FP8 的 safetensors，检查所有权重名、形状、dtype；对单个 Linear 权重反量化后与 BF16 基模型对比误差。

### Phase 7: Qwen3.8 模型构建与前向
1. 实现 `Qwen3ModelBuilder`：根据 `config.json` 与 safetensor 权重名构建完整的 64 层网络。
2. 实现 `Qwen3Model::Forward(input_ids)`：
   - Embedding lookup
   - 64 层循环（DeltaNet × 3 + GQA Attention × 1 为一组，共 16 组）
   - RMSNorm + LM Head
   - Sampling（greedy / top-k / top-p）
3. 支持短上下文（4k / 8k）先跑通。
4. **验证**：与 Transformers 在短 prompt 上输出一致。

### Phase 8: 多设备切分与流水线
1. 实现 `PipelineStage`：持有子图、目标设备、输入输出 buffer。
2. 实现 `StageConnector`：管理相邻 stage 之间的 `d2d` 或 `d2h/h2d` 传输。
3. 实现 `PipelineExecutor`：按拓扑顺序执行 stages，使用 Event 同步。
4. 实现切分策略：
   - 按层数切分到 iGPU/dGPU
   - 由于模型 27.8GB 远大于单卡显存，必要时引入 CPU offload
5. **验证**：在两设备上运行部分层，中间传输正确，输出与单设备一致。

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

## 下一步建议
确认是否按此计划开始实施 **Phase 0 脚手架与构建系统**。由于目标模型规模较大，建议先从 CPU-only 最小闭环开始，再逐步加入 HIP 后端与 FP8 处理。
