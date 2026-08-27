// qwen_infer.cc
// 真实 Qwen3.8 推理 demo（骨架）。
// 目标：加载真实 safetensor 权重，把 prompt 字符串转成 token ids，跑完整/单层
// 模型前向，再把生成的 token ids 解码回文本。
//
// 当前实现重点暴露接口缺口：
//  - 没有 tokenizer，因此文本 <-> ids 需要外部处理；
//  - 没有完整 Qwen3Model / GPU 端到端前向；
//  - 没有量化权重 -> GEMM 的直接路径（需要先把 FP8 反量化到 FP32）。
//
// 用法（当前为占位运行）：
//   qwen_infer.exe <model_dir> [backend=cpu|hip] [max_new_tokens=20]
//
// 它会验证模型目录、加载 config、打印权重/层信息，并提示还缺什么组件。
// compile: Set-Location d:\Vicold\Vicold.HybridAI; cmake --build build-debug --target qwen_infer --config Debug
// $env:HIP_VISIBLE_DEVICES="1"
// $env:PATH = "C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin;" + $env:PATH
// run: .\build-debug\bin\Debug\qwen_infer.exe 'E:\models\Qwen3.8-27B-FP8' cpu

//C:\Users\yinxi\.venv\Scripts\Activate.ps1; $env:HIP_VISIBLE_DEVICES="1"; $env:PATH="C:\Users\yinxi\.venv\Lib\site-packages\_rocm_sdk_devel\bin;" + $env:PATH; d:\Vicold\Vicold.HybridAI\build-debug\bin\Debug\qwen_infer.exe E:/models/Qwen3.8-27B-FP8 gpu 1

#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/device_manager.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"
#include "core/tensor.h"
#include "io/safetensor_loader.h"
#include "models/qwen3_config.h"
#include "models/qwen3_weights.h"
#include "ops/fp8_dequant.h"
#include "tokenizer/qwen_tokenizer.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/softmax.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace hybridai;
using namespace hybridai::models;
using namespace hybridai::ops;

namespace {

void print_usage(const char* program) {
    std::cout << "Usage: " << program
              << " <model_dir> [backend=cpu|hip] [max_new_tokens]"
              << std::endl;
    std::cout << "Example:\n  " << program
              << " E:/models/Qwen3.8-27B-FP8 cpu 20" << std::endl;
}

// 从文件读取 prompt 文本；如果文件不存在就把参数本身当 prompt。
std::string load_prompt_text(const std::string& path_or_text) {
    std::ifstream file(path_or_text);
    if (!file) {
        return path_or_text;
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                         std::istreambuf_iterator<char>());
}

#ifdef HYBRIDAI_HAS_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

// Tokenizer 实例，由 main 初始化后传入。
const hybridai::tokenizer::QwenTokenizer* g_tokenizer = nullptr;

namespace {

// 临时探测 HIP 设备与 rocBLAS 状态，帮助排查为什么 gpu 参数 fallback 到 cpu。
void probe_hip_devices() {
#ifdef HYBRIDAI_HAS_HIP
    int count = 0;
    hipError_t herr = hipGetDeviceCount(&count);
    std::cout << "[HIP probe] hipGetDeviceCount -> " << herr
              << " (" << hipGetErrorString(herr) << "), count=" << count
              << std::endl;
    for (int id = 0; id < count; ++id) {
        hipDeviceProp_t props = {};
        herr = hipGetDeviceProperties(&props, id);
        std::cout << "  device " << id << ": " << props.name
                  << ", arch=" << props.gcnArchName
                  << ", integrated=" << props.integrated
                  << ", hipGetDeviceProperties=" << herr << std::endl;

        herr = hipSetDevice(id);
        if (herr != hipSuccess) {
            std::cout << "    hipSetDevice failed: " << hipGetErrorString(herr)
                      << std::endl;
            continue;
        }
        rocblas_handle handle = nullptr;
        rocblas_status rstatus = rocblas_create_handle(&handle);
        std::cout << "    rocblas_create_handle -> " << rstatus;
        if (rstatus == rocblas_status_success && handle != nullptr) {
            std::cout << " (ok)";
            rocblas_destroy_handle(handle);
        } else {
            const char* msg = rocblas_status_to_string(rstatus);
            std::cout << " (" << (msg ? msg : "unknown") << ")";
        }
        std::cout << std::endl;
    }
#else
    std::cout << "[HIP probe] HYBRIDAI_HAS_HIP not defined (HIP backend not "
                 "compiled in)"
              << std::endl;
#endif
}

} // namespace

std::vector<int64_t> tokenize(const std::string& text) {
    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {
        return g_tokenizer->encode(text, true);
    }
    std::cerr << "[WARN] Tokenizer not loaded. "
              << "Using placeholder token ids." << std::endl;
    return {151644, 872, 198, 151645};
}

std::string decode_ids(const std::vector<int64_t>& ids) {
    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {
        return g_tokenizer->decode(ids, false);
    }
    std::string out = "<decoded> ";
    for (int64_t id : ids) {
        out += std::to_string(id) + " ";
    }
    out += "</decoded>";
    return out;
}

// 选择可用设备
// "gpu" 是通用别名，映射到已注册的 GPU 后端（当前为 hip）。
Device select_device(const std::string& backend_name) {
    DeviceManager::instance().initialize();
    Device chosen = Device::Cpu();

    std::vector<std::string> candidates;
    if (backend_name == "gpu") {
        candidates = {"hip", "cuda"};
    } else {
        candidates.push_back(backend_name);
    }

    for (const std::string& name : candidates) {
        for (const Device& d : DeviceManager::instance().devices()) {
            if (d.backend() == name && d.is_gpu()) {
                return d;
            }
            if (d.backend() == name) {
                chosen = d;
            }
        }
    }
    return chosen;
}

// 构造一个全 FP32 tensor 并用 val 填充
Tensor make_filled_tensor(const Shape& shape, float val, Backend* backend,
                          const Device& device) {
    MemoryType mem_type = device.is_cpu() ? MemoryType::Host
                                            : MemoryType::Unified;
    size_t bytes = static_cast<size_t>(shape.numel()) * sizeof(float);
    auto buf = backend->create_buffer(bytes, mem_type);
    if (buf == nullptr) {
        return Tensor();
    }
    float* p = static_cast<float*>(buf->data());
    for (int64_t i = 0; i < shape.numel(); ++i) {
        p[i] = val;
    }
    return Tensor(shape, DType::FP32, device, buf);
}

// 对 QuantizedWeight 做最小验证：确认 values/scales 都加载成功
void inspect_quantized_weight(const std::string& name,
                               const QuantizedWeight& qw) {
    std::cout << "  " << name << ": ";
    if (qw.values.buffer() == nullptr) {
        std::cout << "not loaded";
    } else {
        std::cout << "shape=[";
        for (size_t i = 0; i < qw.values.shape().ndim(); ++i) {
            if (i) std::cout << ",";
            std::cout << qw.values.shape().dim(i);
        }
        std::cout << "] dtype=" << static_cast<int>(qw.values.dtype());
        std::cout << " scales=" << (qw.has_scales() ? "yes" : "no");
    }
    std::cout << std::endl;
}

} // namespace

int main(int argc, char* argv[]) {
    // 尽早刷新，确保即使后续崩溃也能看到启动痕迹
    std::cout << "[qwen_infer] enter main, argc=" << argc << std::endl;
    std::cout.flush();

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::cout << "[qwen_infer] initializing backends..." << std::endl;
    std::cout.flush();
    InitializeBuiltinBackends();
    std::cout << "[qwen_infer] backends initialized" << std::endl;
    std::cout.flush();

    probe_hip_devices();

    fs::path model_dir = argv[1];
    std::string backend_name = (argc > 2) ? argv[2] : "cpu";
    int max_new_tokens = (argc > 3) ? std::atoi(argv[3]) : 20;

    std::cout << "========================================" << std::endl;
    std::cout << "Qwen3.8 Inference Demo (skeleton)" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "Backend:   " << backend_name << std::endl;
    std::cout << "Max new tokens: " << max_new_tokens << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. 验证目录存在
    if (!fs::exists(model_dir) || !fs::is_directory(model_dir)) {
        std::cerr << "[ERROR] Model directory does not exist: " << model_dir
                  << std::endl;
        return 1;
    }

    // 2. 加载 config
    Qwen3Config config;
    fs::path config_path = model_dir / "config.json";
    Status status = config.load_json(config_path.string());
    if (!status.ok()) {
        std::cerr << "[ERROR] Failed to load config: " << status.message()
                  << std::endl;
        return 1;
    }
    std::cout << "\n[Qwen3 Config]" << std::endl;
    std::cout << "  hidden_size: " << config.hidden_size << std::endl;
    std::cout << "  num_hidden_layers: " << config.num_hidden_layers
              << std::endl;
    std::cout << "  num_attention_heads: " << config.num_attention_heads
              << std::endl;
    std::cout << "  num_key_value_heads: " << config.num_key_value_heads
              << std::endl;
    std::cout << "  head_dim: " << config.head_dim << std::endl;
    std::cout << "  rope_head_dim: " << config.rope_head_dim << std::endl;
    std::cout << "  intermediate_size: " << config.intermediate_size
              << std::endl;
    std::cout << "  vocab_size: " << config.vocab_size << std::endl;
    std::cout << "  max_position_embeddings: " << config.max_position_embeddings
              << std::endl;
    std::cout << "  rope_theta: " << config.rope_theta << std::endl;

    // 3. 选择设备并创建 backend（先于权重加载，确保 GPU 识别正确）
    Device device = select_device(backend_name);
    std::cout << "\n[Device] backend=" << device.backend()
              << ", id=" << device.id()
              << ", type=" << DeviceTypeToString(device.type())
              << ", unified=" << device.unified_memory_supported()
              << std::endl;

    auto backend = BackendRegistry::instance().create_backend(device);
    if (backend == nullptr || !backend->is_available()) {
        std::cerr << "[ERROR] Backend " << backend_name
                  << " is not available." << std::endl;
        return 1;
    }

    // 4. 打开权重加载器，扫描模型文件
    Qwen3WeightLoader loader;
    status = loader.open(model_dir.string(), config);
    if (!status.ok()) {
        std::cerr << "[ERROR] Failed to open model weights: "
                  << status.message() << std::endl;
        return 1;
    }
    std::cout << "\n[Weight Loader] Opened successfully." << std::endl;

    // 5. 加载共享权重（embedding / final norm / lm_head）并做最小检查
    Qwen3SharedWeights shared;
    status = loader.load_shared(device, &shared);
    if (!status.ok()) {
        std::cerr << "[ERROR] Failed to load shared weights: "
                  << status.message() << std::endl;
        return 1;
    }
    std::cout << "\n[Shared Weights]" << std::endl;
    std::cout << "  embed_tokens: "
              << (shared.embed_tokens.buffer() != nullptr ? "loaded"
                                                             : "missing")
              << std::endl;
    std::cout << "  final_norm: "
              << (shared.final_norm.buffer() != nullptr ? "loaded" : "missing")
              << std::endl;
    std::cout << "  lm_head: "
              << (shared.lm_head.buffer() != nullptr ? "loaded" : "missing")
              << std::endl;

    // 6. 加载前 4 层做结构探查（临时加速测试，避免加载全部 64 层）
    constexpr int64_t k_probe_layers = 4;
    std::vector<Qwen3LayerWeights> probe_layers;
    for (int64_t li = 0; li < k_probe_layers; ++li) {
        Qwen3LayerWeights layer;
        status = loader.load_layer(li, device, &layer);
        if (!status.ok()) {
            std::cerr << "[ERROR] Failed to load layer " << li << ": "
                      << status.message() << std::endl;
            return 1;
        }
        std::cout << "\n[Layer " << li << "] attention="
                  << layer.is_attention_layer << std::endl;
        inspect_quantized_weight("q_proj", layer.q_proj);
        inspect_quantized_weight("k_proj", layer.k_proj);
        inspect_quantized_weight("v_proj", layer.v_proj);
        inspect_quantized_weight("o_proj", layer.o_proj);
        inspect_quantized_weight("mlp_gate_proj", layer.mlp_gate_proj);
        inspect_quantized_weight("mlp_up_proj", layer.mlp_up_proj);
        inspect_quantized_weight("mlp_down_proj", layer.mlp_down_proj);
        probe_layers.push_back(std::move(layer));
    }
    std::cout << "\n[Probe layers loaded: " << probe_layers.size() << "]"
              << std::endl;

    // 0. 加载 tokenizer（如果有 tokenizer.json）
    hybridai::tokenizer::QwenTokenizer tokenizer;
    Status tok_status = tokenizer.load(model_dir.string());
    if (tok_status.ok()) {
        g_tokenizer = &tokenizer;
        std::cout << "\n[Tokenizer] Loaded successfully." << std::endl;
    } else {
        std::cout << "\n[Tokenizer] Not loaded: " << tok_status.message()
                  << std::endl;
    }

    // 7. 文本 -> ids
    std::string prompt = "Hello, how are you?";
    std::vector<int64_t> input_ids = tokenize(prompt);
    std::cout << "\n[Prompt -> Token IDs]" << std::endl;
    std::cout << "  prompt: " << prompt << std::endl;
    std::cout << "  ids:    [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << input_ids[i];
    }
    std::cout << "]" << std::endl;

    // 8. 展示一个最小可运行链路：embedding lookup + LM head（随机/占位权重）
    // 真实实现需要：
    //   - 从 embed_tokens.weight 中按 input_ids 索引出 [seq_len, hidden_size]
    //   - 跑 64 层 DeltaNet/Attention
    //   - 用 lm_head 投影到 vocab，采样下一个 token
    // 这里用随机数据验证 Linear -> RMSNorm -> Softmax 能在 device 上跑通。
    std::cout << "\n[Minimal forward smoke test on device]" << std::endl;
    int64_t seq_len = static_cast<int64_t>(input_ids.size());
    Tensor hidden = make_filled_tensor(
        Shape{seq_len, config.hidden_size}, 0.01f, backend.get(), device);
    Tensor norm_w = make_filled_tensor(Shape{config.hidden_size}, 1.0f,
                                        backend.get(), device);

    Tensor after_norm = RMSNorm::forward(hidden, norm_w, 1e-6f);
    if (after_norm.buffer() == nullptr) {
        std::cerr << "[ERROR] RMSNorm failed on device." << std::endl;
        return 1;
    }
    std::cout << "  RMSNorm ok: shape=[" << after_norm.shape().dim(0) << ","
              << after_norm.shape().dim(1) << "]" << std::endl;

    // LM head 占位：把 hidden 投影到 vocab_size
    Tensor lm_head_w = make_filled_tensor(
        Shape{config.vocab_size, config.hidden_size}, 0.001f, backend.get(),
        device);
    Tensor logits = Linear::forward(after_norm, lm_head_w, true, nullptr);
    if (logits.buffer() == nullptr) {
        std::cerr << "[ERROR] LM head Linear failed on device." << std::endl;
        return 1;
    }
    std::cout << "  LM head Linear ok: shape=[" << logits.shape().dim(0) << ","
              << logits.shape().dim(1) << "]" << std::endl;

    Tensor probs = Softmax::forward(logits);
    if (probs.buffer() == nullptr) {
        std::cerr << "[ERROR] Softmax failed on device." << std::endl;
        return 1;
    }
    std::cout << "  Softmax ok: shape=[" << probs.shape().dim(0) << ","
              << probs.shape().dim(1) << "]" << std::endl;

    // 9. 简单贪心采样（取最后一个位置最大概率的 id）
    int64_t next_id = 0;
    {
        const float* last_row = static_cast<const float*>(probs.data()) +
                                (probs.shape().dim(0) - 1) *
                                    probs.shape().dim(1);
        float best = -1.0f;
        for (int64_t i = 0; i < probs.shape().dim(1); ++i) {
            if (last_row[i] > best) {
                best = last_row[i];
                next_id = i;
            }
        }
    }
    std::cout << "\n[Greedy next token] id=" << next_id << std::endl;

    // 10. ids -> 文本（当前为占位）
    std::vector<int64_t> output_ids(input_ids);
    output_ids.push_back(next_id);
    std::string output_text = decode_ids(output_ids);
    std::cout << "\n[Generated text]\n" << output_text << std::endl;

    // 11. 同步并总结缺失项
    status = backend->synchronize();
    if (!status.ok()) {
        std::cerr << "[ERROR] Backend synchronize failed: "
                  << status.message() << std::endl;
        return 1;
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Demo finished. Missing pieces for real inference:"
              << std::endl;
    std::cout << "  1. Tokenizer (text <-> ids) implementation." << std::endl;
    std::cout << "  2. Qwen3Model full forward with 64 layers." << std::endl;
    std::cout << "  3. QuantizedWeight -> FP32 dequant + GEMM path."
              << std::endl;
    std::cout << "  4. KV cache management and autoregressive sampling."
              << std::endl;
    std::cout << "  5. Chat template / enable_thinking prompt wrapping."
              << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}