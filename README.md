# Vicold.HybridAI

跨平台 C++ AI 推理引擎，目标支持 ROCm/HIP 与 CUDA 后端，业务代码与原生 GPU API 完全隔离，便于后续接入手写 kernel（Tile 路线）。

## 项目概况

- **语言与标准**：C++20，CMake 3.20+
- **后端抽象**：`Backend` 接口统一封装 GEMM、内存拷贝、同步等操作；只有 `src/backends/<backend>/` 内部可以调用原生 API。
- **当前实现后端**：
  - `cpu`：纯 CPU 参考实现，用于测试与诊断 fallback。
  - `hip`：ROCm/HIP 后端，基于 rocBLAS 提供 FP32 GEMM，后续预留手写 kernel 替换入口。
- **当前目标模型**：Qwen3.5/Qwen3.8 27B BF16 文本模型；FP8 支持保留为后续扩展。
- **主要模块**：
  - `src/core/`：Tensor、DType、Shape、Device、Status。
  - `src/memory/`：Allocator、MemoryPool、MemoryPlanner。
  - `src/backends/`：后端接口与 HIP/CPU 实现。
  - `src/ops/`：Linear、RMSNorm、RoPE、Softmax、Activation、GQA Attention、DeltaNet、FP8 反量化。
  - `src/io/`：safetensors 加载与 index 分片支持。
  - `src/models/`：Qwen3 配置、权重 schema、权重放置策略。
  - `src/cli/`：`hybridai` 命令行工具。
  - `tests/`：GoogleTest 单元与集成测试。
  - `doc/`：项目计划、架构与适配笔记。

## 目录结构

```text
Vicold.HybridAI/
├── CMakeLists.txt              # 根构建入口
├── cmake/                      # 构建选项与 ROCm 探测
│   ├── Options.cmake
│   ├── hip-windows-toolchain.cmake
│   └── ...
├── src/
│   ├── backends/               # 后端抽象与 HIP/CPU 实现
│   ├── cli/                    # 命令行入口
│   ├── core/                   # Tensor / Device / DType / Status
│   ├── io/                     # safetensors 加载
│   ├── memory/                 # 分配器与内存规划
│   ├── models/                 # Qwen3 配置与权重
│   └── ops/                    # 算子实现与 KernelRegistry
├── tests/                      # GTest 测试
├── doc/                        # 项目文档
└── envdevice.md                # 当前开发机 ROCm 环境记录
```

## 构建要求

- Windows 10/11 或 Linux
- CMake 3.20+
- Visual Studio 2026（当前开发使用）或 Visual Studio 2022+（Windows）；GCC/Clang（Linux）
- 可选：ROCm/HIP SDK（Windows 上推荐 Python venv `_rocm_sdk_devel` 包）

## 构建命令

### 当前推荐流程：Linux + HIP

当前默认配置只构建 HIP 版共享库，不构建 demo、CLI 和单元测试：

```text
HYBRIDAI_ENABLE_HIP=ON
HYBRIDAI_ENABLE_CPU=OFF
BUILD_SHARED_LIBS=ON
HYBRIDAI_BUILD_TESTS=OFF
HYBRIDAI_BUILD_CLI=OFF
HYBRIDAI_BUILD_DEMOS=OFF
```

`nlohmann_json` 已放在 `third_party/nlohmann_json`，配置过程不需要从
GitHub 下载该依赖。

#### 1. 构建 `libhybridai.so`

在项目根目录创建并进入 `build`：

```bash
mkdir -p build
cd build
cmake ..
make -j$(nproc)
```

也可以使用与生成器无关的写法：

```bash
cmake -S . -B build
cmake --build build --parallel
```

成功后生成：

```text
build/libhybridai.so
```

如需同时启用 CPU 参考后端：

```bash
cmake -S . -B build -DHYBRIDAI_ENABLE_CPU=ON
cmake --build build --parallel
```

#### 2. 使用已编译的 `.so` 独立构建 demo

`demo/CMakeLists.txt` 可以作为独立 CMake 项目使用，不会重新编译
`hybridai`，默认链接 `build/libhybridai.so`：

```bash
cmake -S demo -B demo/build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc

cmake --build demo/build --target qwen_infer -j$(nproc)
```

生成文件：

```text
demo/build/bin/qwen_infer
```

如果动态库不在默认位置，通过 `HYBRIDAI_LIBRARY` 指定：

```bash
cmake -S demo -B demo/build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DHYBRIDAI_LIBRARY=/absolute/path/to/libhybridai.so
```

如果 demo 与 HybridAI 源码不在同一个仓库目录，还需要指定源码根目录：

```bash
cmake -S demo -B demo/build \
  -DCMAKE_CXX_COMPILER=/opt/dtk/bin/hipcc \
  -DHYBRIDAI_ROOT=/absolute/path/to/Vicold.HybridAI \
  -DHYBRIDAI_LIBRARY=/absolute/path/to/libhybridai.so
```

同一个独立 demo 工程也可以构建其他示例：

```bash
cmake --build demo/build --target simple_infer -j$(nproc)
cmake --build demo/build --target test_tokenizer -j$(nproc)
```

#### 3. 检查动态链接

```bash
ldd demo/build/bin/qwen_infer | grep -E 'hybridai|rocblas|galaxyhip'
```

输出应包含 `build/libhybridai.so`、`librocblas.so` 和 HIP runtime。
独立 demo 已设置构建 RPATH，正常情况下不需要手动设置
`LD_LIBRARY_PATH`。

#### 4. 启动检查

不传模型路径时，程序应打印用法并退出：

```bash
./demo/build/bin/qwen_infer
```

预期输出开头：

```text
[qwen_infer] enter main, argc=1
Usage: ./demo/build/bin/qwen_infer <model_dir> [backend=cpu|hip] [max_devices=8]
```

#### 5. 运行真实 Qwen 模型

```bash
./demo/build/bin/qwen_infer \
  /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  hip 8
```

参数含义：

- 第一个参数：模型快照目录；
- 第二个参数：后端，当前使用 `hip`；
- 第三个参数：最多使用的设备数，当前八卡环境使用 `8`。

该流程会验证 tokenizer、分片 SafeTensor 权重加载、BF16 投影、真实
DeltaNet/reference forward、完整 64 层前向、LM head 和 greedy decode。

#### 6. 可选：构建单元测试

单元测试默认关闭。需要测试时重新配置独立构建目录：

```bash
cmake -S . -B build-tests \
  -DHYBRIDAI_ENABLE_HIP=ON \
  -DHYBRIDAI_ENABLE_CPU=ON \
  -DHYBRIDAI_BUILD_TESTS=ON

cmake --build build-tests --target hybridai_test --parallel
ctest --test-dir build-tests --output-on-failure
```

注意：测试依赖 GoogleTest。如果系统没有可用的 GoogleTest，且仓库中
没有准备 `third_party/deps_local/googletest-1.15.2`，CMake 会尝试从
GitHub 获取。当前主要端到端验证方式是直接运行 `qwen_infer`。

### Windows 默认构建（CPU + 测试）

```powershell
cmake -B build -G "Visual Studio 18 2026" -A x64 `
  -D HYBRIDAI_ENABLE_CPU=ON `
  -D HYBRIDAI_BUILD_TESTS=ON `
  -D HYBRIDAI_BUILD_CLI=ON

cmake --build build --config Release
```

### Windows HIP/ROCm 构建

开发机 ROCm SDK 位于 `C:/Users/yinxi/.venv/Lib/site-packages/_rocm_sdk_devel`，可直接使用工具链预设：

```powershell
cmake -B build-debug -G "Visual Studio 18 2026" -A x64 `
  -C cmake/hip-windows-toolchain.cmake

cmake --build build-debug --config Debug --target hybridai_test hybridai_cli simple_infer
```

或手动指定 ROCm 路径：

```powershell
cmake -B build-debug -G "Visual Studio 18 2026" -A x64 `
  -D HYBRIDAI_ENABLE_HIP=ON `
  -D HYBRIDAI_ENABLE_CPU=ON `
  -D HYBRIDAI_BUILD_TESTS=ON `
  -D HYBRIDAI_BUILD_CLI=ON `
  -D HYBRIDAI_ROCM_ROOT="C:/Users/yinxi/.venv/Lib/site-packages/_rocm_sdk_devel"

cmake --build build-debug --config Debug --target hybridai_test hybridai_cli simple_infer
```

> Windows 上 `hip_backend.cc` 使用 MSVC 直接编译 HIP host API，不经过 ROCm Clang，从而避免 MSVC Debug CRT 与 ROCm Clang runtime library 不匹配问题。

### Linux HIP/ROCm 构建

标准 ROCm 安装（通常位于 `/opt/rocm`）可直接构建：

```bash
cmake -B build \
  -G Ninja \
  -D HYBRIDAI_ENABLE_HIP=ON \
  -D HYBRIDAI_ENABLE_CPU=ON \
  -D HYBRIDAI_BUILD_TESTS=ON \
  -D HYBRIDAI_BUILD_CLI=ON

cmake --build build
```

当前 DTK 环境安装在 `/opt/dtk`。`nlohmann_json` 已随仓库提供；离线环境
可以关闭依赖 fmt/spdlog/GTest 的 CLI 与测试，构建核心库和 demo：

```bash
cmake -S . -B build-linux-hip -G Ninja \
  -D CMAKE_BUILD_TYPE=Debug \
  -D HYBRIDAI_ROCM_ROOT=/opt/dtk \
  -D HYBRIDAI_ENABLE_HIP=ON \
  -D HYBRIDAI_ENABLE_CPU=ON \
  -D HYBRIDAI_BUILD_TESTS=OFF \
  -D HYBRIDAI_BUILD_CLI=OFF \
  -D HYBRIDAI_BUILD_DEMOS=ON

cmake --build build-linux-hip --parallel
./build-linux-hip/bin/simple_infer hip
```

### Linux 八卡 BF16 模型加载

当前 `qwen_infer` 支持把 Qwen3.8/Qwen3.5 27B 的文本模型权重直接加载到
最多八张非统一内存 HIP GPU。64 层按连续区间切分，embedding 放在首卡，
final norm 和 LM head 放在末卡；BF16 权重保持原始精度并使用设备显存。

```bash
./build-linux-hip/bin/qwen_infer \
  /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master \
  hip 8
```

在当前 8 × 64 GiB `gfx936` 环境实测加载成功：文本模型权重约
`50.10 GiB`，首末卡各约 `8.04 GiB`，其余卡各约 `5.67 GiB`。
该命令可以继续执行 BF16 投影、真实 DeltaNet/reference forward、完整
64 层前向、LM head 与 greedy decode。当前待改进项主要是 tokenizer
字节级解码质量，以及将 host-reference DeltaNet 替换为优化 HIP kernel。

`HYBRIDAI_ROCM_ROOT` 支持标准 ROCm 与 DTK 这类派生 SDK；构建逻辑会在
SDK 根目录下查找 `hip`、`rocblas`、`dcc` 和 `dcc/comgr` 的 CMake 包。
网络可用或 `third_party/deps_local/` 已准备完整依赖时，可重新启用 CLI
与测试。Windows 仍沿用下文的 Visual Studio + HIP host API 构建方案。

## 运行环境配置（Windows HIP）

当前开发机有两块 AMD GPU：

- `device 0`：AMD Radeon RX 5700，`gfx1010`，ROCm SDK 未提供对应 Tensile kernel。
- `device 1`：AMD Radeon 890M Graphics，`gfx1150`，SDK 提供对应 kernel 数据。

因此运行 HIP 程序前需要过滤可见设备并指定 Tensile 库路径：

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
$env:ROCBLAS_TENSILE_LIBPATH = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'
```

如果目录中只有 `TensileLibrary_lazy_gfx1150.dat`，可复制一份重命名为 `TensileLibrary.dat`：

```powershell
$lib = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'
Copy-Item "$lib\TensileLibrary_lazy_gfx1150.dat" "$lib\TensileLibrary.dat" -ErrorAction SilentlyContinue
```

## 测试流程

### 列出可用设备

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
$env:ROCBLAS_TENSILE_LIBPATH = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'

.\build-debug\bin\Debug\hybridai.exe devices
```

预期输出示例：

```text
Available devices:
  device#0: backend=cpu, type=cpu, unified_memory=false
  device#0: backend=hip, type=igpu, unified_memory=true
```

### 运行单元测试

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
$env:ROCBLAS_TENSILE_LIBPATH = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'

.\build-debug\tests\Debug\hybridai_test.exe
```

### 只运行 HIP GEMM 测试

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
$env:ROCBLAS_TENSILE_LIBPATH = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'

.\build-debug\tests\Debug\hybridai_test.exe --gtest_filter='BackendComputeTest.HipFp32GemmMatchesReference'
```

### 运行最小推理 demo

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
$env:ROCBLAS_TENSILE_LIBPATH = 'C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin\rocblas\library\gfx1150'

# CPU 后端（默认）
.\build-debug\bin\Debug\simple_infer.exe cpu

# HIP 后端
.\build-debug\bin\Debug\simple_infer.exe hip
```

`demo/simple_infer.cc` 是一个不依赖 GTest 的最小推理示例，依次执行 `Linear -> RMSNorm -> Softmax`，并校验 Softmax 输出为合法概率分布。

## 后端封装规范

业务代码（`src/core`、`src/memory`、`src/ops`、`src/runtime`、`src/models`、`src/io`、`src/cli`）禁止直接包含或调用以下原生 API：

- `<cuda_runtime.h>` / `<hip/hip_runtime.h>`
- `<cublas_v2.h>` / `<rocblas.h>`
- `hipMalloc`、`cudaMalloc`、`hipMemcpy`、`cudaMemcpy` 等

所有硬件访问都通过 `src/backends/interface/backend.h` 中定义的 `Backend`、`Stream`、`Event`、`Buffer`、`Allocator` 抽象完成。

## 当前进展

- ✅ CMake/C++20 工程脚手架、核心抽象、内存管理、后端注册表。
- ✅ CPU 后端与基础算子（Linear、RMSNorm、RoPE、Softmax、Elementwise、FP8 反量化）。
- ✅ Gated GQA Attention 与 Gated DeltaNet 参考实现。
- ✅ safetensors 单文件与分片加载、Qwen3 配置解析、权重 schema 与放置规划。
- ✅ Windows HIP/ROCm 构建链路打通：MSVC 直接编译 HIP host API，Debug 配置链接成功。
- ✅ 真实 GPU（gfx1150）上 rocBLAS FP32 GEMM 测试通过。
- ✅ Linux DTK/HIP 构建链路打通：GCC + Ninja 成功构建核心库和 demo，`simple_infer hip` 已在 `gfx936` 上完成 `Linear -> RMSNorm -> Softmax`。
- ✅ 8×`gfx936` 非统一内存环境中，约 50.10 GiB BF16 文本权重已按连续层完整常驻设备。
- ✅ 真实 BF16 投影、DeltaNet/reference forward、64 层前向、LM head 和 greedy decode 已跑通。
- ✅ `demo/CMakeLists.txt` 可独立链接预编译的 `libhybridai.so` 构建示例。
- ✅ 新增 `demo/simple_infer.cc`：不依赖测试框架的最小推理用例，支持 CPU/HIP 后端。
- 🔄 进行中：tokenizer 字节级解码修正、GPU 基础 kernel 和优化版 HIP DeltaNet。

## 文档索引

- `doc/plan.md`：项目总体计划与阶段化实施状态。
- `doc/architecture.md`：架构设计（待完善）。
- `doc/cross_platform_notes.md`：跨平台构建与封装规范。
- `doc/qwen3_8_27b_fp8.md`：Qwen3.8-27B-FP8 模型适配说明。
- `envdevice.md`：当前开发机 ROCm/HIP 环境记录。
