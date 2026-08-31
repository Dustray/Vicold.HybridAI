// qwen_infer.cc
// Qwen3.5/Qwen3.8 27B BF16 多卡加载与推理入口。
//
// 当前 HIP 路径已将完整文本权重按连续层加载到最多八张离散 GPU；后续
// 在此入口接入 BF16 kernel、partition activation 传输和 prefill/decode。
//
// 编译：cmake --build .. --target qwen_infer --parallel
//
// 用法：qwen_infer <model_dir> [backend=cpu|hip] [max_devices=8]
//                  [--quiet-decode]
// 运行示例（在项目根目录执行）：
// LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}" ./demo/build/qwen_infer /path/to/Qwen3.8-27B hip 8 --quiet-decode
// HIP_ALLOC_INITIALIZE=0 HIP_VISIBLE_DEVICES=2 ./qwen_infer /public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master/ hip 1 --quiet-decode

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
#include "ops/elementwise.h"
#include "ops/attention.h"
#include "ops/delta_net.h"
#include "tokenizer/qwen_tokenizer.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/softmax.h"

#include <cstddef>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <limits>
#include <algorithm>
#include <string>
#include <vector>
#include <cmath>

#ifdef HYBRIDAI_HAS_HIP
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#endif

namespace fs = std::filesystem;
using namespace hybridai;
using namespace hybridai::models;
using namespace hybridai::ops;

namespace {

void print_tensor_stats(const char* label, const Tensor& tensor,
                        Backend* backend, size_t max_values = 0) {
    if (backend == nullptr || tensor.buffer() == nullptr ||
        tensor.dtype() != DType::FP32) {
        return;
    }
    const size_t count = static_cast<size_t>(tensor.numel());
    std::vector<float> values(count);
    if (!backend->memcpy_d2h(values.data(), tensor.buffer().get(),
                             tensor.nbytes()).ok() || values.empty()) {
        return;
    }
    double sum = 0.0;
    double sum_sq = 0.0;
    float min_value = std::numeric_limits<float>::infinity();
    float max_value = -std::numeric_limits<float>::infinity();
    size_t non_finite = 0;
    for (float value : values) {
        if (!std::isfinite(value)) {
            ++non_finite;
            continue;
        }
        sum += value;
        sum_sq += static_cast<double>(value) * value;
        min_value = std::min(min_value, value);
        max_value = std::max(max_value, value);
    }
    const size_t finite = count - non_finite;
    std::cout << "[Stats] " << label << " count=" << count
              << " mean=" << (finite ? sum / finite : 0.0)
              << " rms=" << (finite ? std::sqrt(sum_sq / finite) : 0.0)
              << " min=" << (finite ? min_value : 0.0f)
              << " max=" << (finite ? max_value : 0.0f)
              << " non_finite=" << non_finite;
    if (max_values > 0) {
        std::cout << " first=";
        for (size_t i = 0; i < std::min(max_values, count); ++i) {
            if (i != 0) std::cout << ",";
            std::cout << values[i];
        }
    }
    std::cout << std::endl;
}

void print_usage(const char* program) {
    std::cout << "Usage: " << program
                  << " <model_dir> [backend=cpu|hip] [max_devices=8] "
                      "[--quiet-decode]"
              << std::endl;
    std::cout << "Example:\n  " << program
                  << " /path/to/Qwen3.8-27B hip 8 --quiet-decode" << std::endl;
}

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

std::string decode_ids(const std::vector<int64_t>& ids,
                       bool skip_special_tokens = false) {
    if (g_tokenizer != nullptr && g_tokenizer->is_loaded()) {
        return g_tokenizer->decode(ids, skip_special_tokens);
    }
    std::string out = "<decoded> ";
    for (int64_t id : ids) {
        out += std::to_string(id) + " ";
    }
    out += "</decoded>";
    return out;
}

Status lookup_embedding(const Tensor& embedding, const std::vector<int64_t>& ids,
                        Backend* backend, const Device& device,
                        Tensor* output) {
    if (output == nullptr || backend == nullptr ||
        embedding.buffer() == nullptr ||
        embedding.dtype() != DType::BF16 ||
        embedding.shape().ndim() != 2) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid BF16 embedding lookup arguments");
    }

    const int64_t vocab_size = embedding.shape().dim(0);
    const int64_t hidden_size = embedding.shape().dim(1);
    for (int64_t id : ids) {
        if (id < 0 || id >= vocab_size) {
            return Status(StatusCode::InvalidArgument,
                          "Token id is outside embedding vocabulary");
        }
    }

    const size_t row_bytes =
        static_cast<size_t>(hidden_size) * SizeOfDType(DType::BF16);
    const size_t output_bytes = ids.size() * row_bytes;
    auto buffer =
        backend->create_buffer(output_bytes, MemoryType::Device);
    if (buffer == nullptr) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate embedding output");
    }

    Status status;
    if (device.is_gpu()) {
        auto ids_buffer = backend->create_buffer(
            ids.size() * sizeof(int64_t), MemoryType::Device);
        if (ids_buffer == nullptr) {
            return Status(StatusCode::OutOfMemory,
                          "Failed to allocate embedding ids");
        }
        status = backend->memcpy_h2d(ids_buffer.get(), ids.data(),
                                     ids.size() * sizeof(int64_t));
        if (!status.ok()) return status;
        status = backend->embedding_gather(
            buffer.get(), embedding.buffer().get(), ids_buffer.get(),
            DType::BF16, static_cast<int64_t>(ids.size()), vocab_size,
            hidden_size);
        if (!status.ok()) return status;
    } else {
        const auto* source = static_cast<const uint16_t*>(embedding.data());
        auto* destination = static_cast<uint16_t*>(buffer->data());
        for (size_t row = 0; row < ids.size(); ++row) {
            std::memcpy(destination + row * static_cast<size_t>(hidden_size),
                        source + static_cast<size_t>(ids[row]) * hidden_size,
                        row_bytes);
        }
    }

    *output = Tensor(
        Shape{static_cast<int64_t>(ids.size()), hidden_size}, DType::BF16,
        device, std::move(buffer));
    return Status::OK();
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

std::vector<Device> select_gpu_devices(const std::string& backend_name,
                                       int max_devices) {
    DeviceManager::instance().initialize();
    std::vector<Device> devices;
    for (const Device& device : DeviceManager::instance().devices()) {
        const bool matches = backend_name == "gpu"
                                 ? device.is_gpu()
                                 : device.backend() == backend_name;
        if (device.is_gpu() && matches) {
            devices.push_back(device);
            if (static_cast<int>(devices.size()) >= max_devices) break;
        }
    }
    return devices;
}

void print_device_memory(const std::vector<Device>& devices) {
#ifdef HYBRIDAI_HAS_HIP
    constexpr double gib = 1024.0 * 1024.0 * 1024.0;
    for (const Device& device : devices) {
        if (device.backend() != "hip" ||
            hipSetDevice(device.id()) != hipSuccess) {
            continue;
        }
        size_t free_bytes = 0;
        size_t total_bytes = 0;
        if (hipMemGetInfo(&free_bytes, &total_bytes) == hipSuccess) {
            std::cout << "  hip:" << device.id()
                      << " free=" << free_bytes / gib
                      << " GiB, total=" << total_bytes / gib << " GiB"
                      << std::endl;
        }
    }
#else
    (void)devices;
#endif
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

void inspect_tensor(const std::string& name, const Tensor& tensor) {
    std::cout << "  " << name << ": ";
    if (tensor.buffer() == nullptr) {
        std::cout << "not loaded" << std::endl;
        return;
    }
    std::cout << "shape=[";
    for (size_t i = 0; i < tensor.shape().ndim(); ++i) {
        if (i) std::cout << ",";
        std::cout << tensor.shape().dim(i);
    }
    std::cout << "] dtype=" << static_cast<int>(tensor.dtype()) << std::endl;
}

Tensor qwen_rmsnorm_reference(const Tensor& input, const Tensor& weight,
                              Backend* backend, const Device& device,
                              float eps = 1e-6f);
Tensor make_residual(const Tensor& lhs, const Tensor& rhs, Backend* backend,
                     const Device& device);
Tensor qwen_mlp_reference(const Tensor& input, const Qwen3LayerWeights& layer,
                          Backend* backend, const Device& device);
Tensor qwen_attention_reference(const Tensor& input,
                                const Qwen3LayerWeights& layer,
                                Backend* backend, const Device& device,
                                const Qwen3Config& config,
                                AttentionKVCache* cache = nullptr,
                                int64_t max_cache_len = 0);
Tensor qwen_deltanet_reference(const Tensor& input,
                               const Qwen3LayerWeights& layer,
                               Backend* backend, const Device& device,
                               const Qwen3Config& config,
                               DeltaNetCache* cache = nullptr);
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
    int max_devices = 8;
    bool quiet_decode = false;
    bool max_devices_set = false;
    for (int argument = 3; argument < argc; ++argument) {
        const std::string option = argv[argument];
        if (option == "--quiet-decode") {
            quiet_decode = true;
        } else if (!max_devices_set && !option.empty() && option[0] != '-') {
            max_devices = std::atoi(option.c_str());
            max_devices_set = true;
        } else {
            std::cerr << "[ERROR] Unknown option: " << option << std::endl;
            print_usage(argv[0]);
            return 1;
        }
    }
    if (max_devices <= 0) max_devices = 1;
    if (max_devices > 8) max_devices = 8;

    std::cout << "========================================" << std::endl;
    std::cout << "Qwen3.5/Qwen3.8 BF16 Multi-GPU Loader" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "Backend:   " << backend_name << std::endl;
    std::cout << "Max devices: " << max_devices << std::endl;
    std::cout << "Quiet decode: " << (quiet_decode ? "yes" : "no")
              << std::endl;
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
    std::cout << "  rms_norm_eps: 1e-6, Qwen RMSNorm weight mode: (1 + w)"
              << std::endl;

    // 当前非统一内存环境先执行模型常驻加载。按连续层切分到最多八张卡，
    // embedding 位于首卡，final norm/lm_head 位于末卡。所有权重使用
    // MemoryType::Device，保持原始 BF16，避免加载时膨胀为 FP32。
    if (backend_name == "hip" || backend_name == "gpu") {
        std::vector<Device> devices =
            select_gpu_devices(backend_name, max_devices);
        if (devices.empty()) {
            std::cerr << "[ERROR] No GPU device is available for backend "
                      << backend_name << std::endl;
            return 1;
        }
        std::cout << "\n[Devices] selected " << devices.size() << " GPU(s)"
                  << std::endl;
        print_device_memory(devices);

        Qwen3WeightLoader distributed_loader;
        status = distributed_loader.open(model_dir.string(), config);
        if (!status.ok()) {
            std::cerr << "[ERROR] Failed to open model weights: "
                      << status.message() << std::endl;
            return 1;
        }

        Qwen3DistributedWeights weights;
        status = distributed_loader.load_distributed(devices, &weights);
        if (!status.ok()) {
            std::cerr << "[ERROR] Failed to load distributed weights: "
                      << status.message() << std::endl;
            return 1;
        }

        constexpr double gib = 1024.0 * 1024.0 * 1024.0;
        std::cout << "\n[Distributed Weights] total="
                  << weights.total_weight_bytes / gib << " GiB" << std::endl;
        for (const auto& partition : weights.partitions) {
            std::cout << "  " << partition.device.backend() << ":"
                      << partition.device.id() << " layers "
                      << partition.first_layer << "-" << partition.last_layer
                      << ", weights=" << partition.weight_bytes / gib
                      << " GiB" << std::endl;
        }
        std::cout << "All text-model weights are resident in device memory."
                  << std::endl;

        for (const Qwen3LayerWeights& layer : weights.layers) {
            if (!layer.is_attention_layer) {
                std::cout << "\n[DeltaNet weight layout layer "
                          << layer.layer_index << "]" << std::endl;
                inspect_tensor("linear_attn_norm", layer.linear_attn_norm);
                inspect_tensor("a_log", layer.a_log);
                inspect_tensor("conv1d_weight", layer.conv1d_weight);
                inspect_tensor("dt_bias", layer.dt_bias);
                inspect_tensor("in_proj_a", layer.in_proj_a);
                inspect_tensor("in_proj_b", layer.in_proj_b);
                inspect_quantized_weight("in_proj_qkv", layer.in_proj_qkv);
                inspect_quantized_weight("in_proj_z", layer.in_proj_z);
                inspect_quantized_weight("linear_out_proj", layer.linear_out_proj);
                break;
            }
        }

        hybridai::tokenizer::QwenTokenizer tokenizer;
        Status tok_status = tokenizer.load(model_dir.string());
        if (tok_status.ok()) {
            g_tokenizer = &tokenizer;
            std::cout << "\n[Tokenizer] Loaded successfully." << std::endl;
        } else {
            std::cout << "\n[Tokenizer] Not loaded: "
                      << tok_status.message() << std::endl;
        }
        std::vector<int64_t> eos_token_ids = config.eos_token_ids;
        if (tok_status.ok()) {
            eos_token_ids.insert(eos_token_ids.end(),
                                 tokenizer.eos_token_ids().begin(),
                                 tokenizer.eos_token_ids().end());
        }
        std::sort(eos_token_ids.begin(), eos_token_ids.end());
        eos_token_ids.erase(
            std::unique(eos_token_ids.begin(), eos_token_ids.end()),
            eos_token_ids.end());
        if (eos_token_ids.empty()) {
            std::cerr << "[WARN] No EOS token IDs were found in config.json "
                         "or generation_config.json; generation will stop at "
                         "the token limit."
                      << std::endl;
        } else {
            std::cout << "  eos_token_ids: [";
            for (size_t index = 0; index < eos_token_ids.size(); ++index) {
                if (index != 0) std::cout << ", ";
                std::cout << eos_token_ids[index];
            }
            std::cout << "]" << std::endl;
        }
        // const std::string user_text = "Please explain the principle of quantum entanglement in Chinese. Write at least 500 words.";
        const std::string user_text = "Hello, how are you?";
        std::string prompt;
        if (tok_status.ok()) {
            prompt = tokenizer.build_chat_prompt(
                {{"user", user_text}}, true, false);
        } else {
            prompt =
                "<|im_start|>user\n" + user_text + "<|im_end|>\n"
                "<|im_start|>assistant\n<think>\n\n</think>\n\n";
        }
        std::vector<int64_t> input_ids = tokenize(prompt);
        std::cout << "  user:   " << user_text << std::endl;
        std::cout << "  prompt: " << prompt << std::endl;
        std::cout << "  input_ids: [";
        for (size_t i = 0; i < input_ids.size(); ++i) {
            if (i != 0) std::cout << ", ";
            std::cout << input_ids[i];
        }
        std::cout << "]" << std::endl;

        // 使用真实末层权重验证 typed BF16 GEMM。输入保持为 BF16，输出
        // 由 Linear 使用 FP32 accumulation；这里只验证权重和后端计算，
        // 不宣称已经完成完整 Qwen 前向。
        const Qwen3LayerWeights& last_layer = weights.layers.back();
        const QuantizedWeight* projection =
            last_layer.is_attention_layer ? &last_layer.q_proj
                                           : &last_layer.in_proj_qkv;
        if (projection->values.buffer() == nullptr ||
            projection->values.dtype() != DType::BF16) {
            std::cerr << "[ERROR] Expected a dense BF16 projection in the last "
                         "layer" << std::endl;
            return 1;
        }
        const int64_t projection_rows = projection->values.shape().dim(0);
        const int64_t projection_cols = projection->values.shape().dim(1);
        auto projection_backend =
            BackendRegistry::instance().get_backend(devices.back());
        if (projection_backend == nullptr ||
            !projection_backend->is_available()) {
            std::cerr << "[ERROR] Failed to create backend for the last layer"
                      << std::endl;
            return 1;
        }
        constexpr int64_t smoke_tokens = 1;
        const size_t input_bytes = static_cast<size_t>(smoke_tokens) *
                                   static_cast<size_t>(projection_cols) *
                                   SizeOfDType(DType::BF16);
        auto smoke_input_buffer =
            projection_backend->create_buffer(input_bytes, MemoryType::Device);
        if (smoke_input_buffer == nullptr) {
            std::cerr << "[ERROR] Failed to allocate BF16 GEMM input"
                      << std::endl;
            return 1;
        }
        std::vector<uint16_t> smoke_input(
            static_cast<size_t>(smoke_tokens) *
                static_cast<size_t>(projection_cols),
            0x3F80u);  // BF16(1.0)
        status = projection_backend->memcpy_h2d(
            smoke_input_buffer.get(), smoke_input.data(), input_bytes);
        if (status.ok()) {
            Tensor smoke_input_tensor(
                Shape{smoke_tokens, projection_cols}, DType::BF16,
                devices.back(), std::move(smoke_input_buffer));
            Tensor smoke_output = Linear::forward(
                smoke_input_tensor, projection->values, true, nullptr);
            if (smoke_output.buffer() == nullptr) {
                std::cerr << "[ERROR] Real BF16 projection GEMM failed"
                          << std::endl;
                return 1;
            }
            status = projection_backend->synchronize();
        }
        if (!status.ok()) {
            std::cerr << "[ERROR] Real BF16 projection test failed: "
                      << status.message() << std::endl;
            return 1;
        }
        std::cout << "Real BF16 projection GEMM ok: [1," << projection_cols
                  << "] x [" << projection_rows << "," << projection_cols
                  << "] -> [1," << projection_rows << "]" << std::endl;

        auto embedding_backend =
            BackendRegistry::instance().get_backend(devices.front());
        if (embedding_backend == nullptr ||
            !embedding_backend->is_available()) {
            std::cerr << "[ERROR] Failed to create backend for embedding"
                      << std::endl;
            return 1;
        }
        constexpr int kMaxNewTokens = 1024;
        std::vector<int64_t> generated_ids = input_ids;
        const size_t prompt_token_count = generated_ids.size();
        const int64_t max_cache_len =
            static_cast<int64_t>(input_ids.size()) + kMaxNewTokens;
        if (max_cache_len > config.max_position_embeddings) {
            std::cerr << "[ERROR] Prompt plus generated tokens exceeds model "
                         "context capacity" << std::endl;
            return 1;
        }
        std::vector<AttentionKVCache> attention_caches(
            static_cast<size_t>(config.num_hidden_layers));
        std::vector<DeltaNetCache> deltanet_caches(
            static_cast<size_t>(config.num_hidden_layers));
        using SteadyClock = std::chrono::steady_clock;
        SteadyClock::time_point decode_start;
        int64_t decode_token_count = 0;
        for (int generation_step = 0; generation_step < kMaxNewTokens;
             ++generation_step) {
            const bool is_prefill = generation_step == 0;
            const bool verbose_step = is_prefill || !quiet_decode;
            if (generation_step == 1) decode_start = SteadyClock::now();
            input_ids = generation_step == 0
                ? generated_ids
                : std::vector<int64_t>{generated_ids.back()};
            if (verbose_step) {
                std::cout << (is_prefill ? "[Prefill] tokens="
                                        : "[Decode] tokens=")
                          << input_ids.size() << std::endl;
            }
            Tensor embedded;
        status = lookup_embedding(
            weights.shared.embed_tokens, input_ids,
            embedding_backend.get(), devices.front(), &embedded);
        if (!status.ok()) {
            std::cerr << "[ERROR] Real BF16 embedding lookup failed: "
                      << status.message() << std::endl;
            return 1;
        }
        status = embedding_backend->synchronize();
        if (!status.ok()) {
            std::cerr << "[ERROR] Embedding lookup synchronization failed: "
                      << status.message() << std::endl;
            return 1;
        }

        if (verbose_step) {
            std::cout << "Real BF16 embedding lookup ok: ["
                      << embedded.shape().dim(0) << ","
                      << embedded.shape().dim(1) << "]" << std::endl;
        }

        // 真实前向全程保持模型原生 BF16 tensor；kernel 内部可以使用
        // FP32 累加，但不创建 FP32 activation 或 weight staging tensor。
        const Qwen3LayerWeights& first_layer = weights.layers.front();
        Tensor normalized = qwen_rmsnorm_reference(
            embedded, first_layer.input_layernorm, embedding_backend.get(),
            devices.front(), config.rms_norm_eps);
        if (normalized.buffer() == nullptr) {
            std::cerr << "[ERROR] Real layer RMSNorm failed" << std::endl;
            return 1;
        }
        Tensor linear_output;
        if (first_layer.is_attention_layer) {
            linear_output = qwen_attention_reference(
                normalized, first_layer, embedding_backend.get(), devices.front(),
                config, &attention_caches.front(), max_cache_len);
        } else {
            linear_output = qwen_deltanet_reference(
                normalized, first_layer, embedding_backend.get(), devices.front(),
                config, &deltanet_caches.front());
        }
        if (linear_output.buffer() == nullptr) {
            std::cerr << "[ERROR] Real layer attention/DeltaNet failed" << std::endl;
            return 1;
        }
        Tensor layer0_output = make_residual(
            embedded, linear_output, embedding_backend.get(),
            devices.front());
        if (layer0_output.buffer() == nullptr) {
            std::cerr << "[ERROR] Real layer residual failed" << std::endl;
            return 1;
        }
        status = embedding_backend->synchronize();
        if (!status.ok()) {
            std::cerr << "[ERROR] Real layer synchronization failed: "
                      << status.message() << std::endl;
            return 1;
        }
        if (verbose_step) {
            std::cout << "Real layer 0 attention/DeltaNet path ok: output=["
                      << layer0_output.shape().dim(0) << ","
                      << layer0_output.shape().dim(1) << "]" << std::endl;
        }

        // 补齐第 0 层的 post-mixer RMSNorm、MLP 和第二个 residual。
        // Decoder layer 的两个子层都必须执行，不能只验证第一个 mixer。
        Tensor layer0_post_norm = qwen_rmsnorm_reference(
            layer0_output, first_layer.post_attention_layernorm,
            embedding_backend.get(), devices.front(), config.rms_norm_eps);
        if (layer0_post_norm.buffer() == nullptr) {
            std::cerr << "[ERROR] Layer 0 post-mixer RMSNorm failed"
                      << std::endl;
            return 1;
        }
        Tensor layer0_mlp = qwen_mlp_reference(
            layer0_post_norm, first_layer, embedding_backend.get(),
            devices.front());
        if (layer0_mlp.buffer() == nullptr) {
            std::cerr << "[ERROR] Layer 0 MLP failed" << std::endl;
            return 1;
        }
        Tensor hidden_state = make_residual(
            layer0_output, layer0_mlp, embedding_backend.get(),
            devices.front());
        if (hidden_state.buffer() == nullptr) {
            std::cerr << "[ERROR] Layer 0 MLP residual failed" << std::endl;
            return 1;
        }
        // 继续执行其余层的真实 Attention/DeltaNet、MLP 和残差路径。
        // Prefill 后每一轮只处理最新 token，逐层复用 KV/state cache。
        for (int64_t layer_index = 1;
             layer_index < config.num_hidden_layers; ++layer_index) {
            const Qwen3LayerWeights& layer =
                weights.layers[static_cast<size_t>(layer_index)];
            const Device layer_device =
                weights.layer_devices[static_cast<size_t>(layer_index)];
            auto layer_backend =
                BackendRegistry::instance().get_backend(layer_device);
            if (layer_backend == nullptr || !layer_backend->is_available()) {
                std::cerr << "[ERROR] Failed to create backend for layer "
                          << layer_index << std::endl;
                return 1;
            }
            if (hidden_state.device() != layer_device) {
                hidden_state = hidden_state.to(layer_device);
                if (hidden_state.buffer() == nullptr) {
                    std::cerr << "[ERROR] Failed to transfer activation to layer "
                              << layer_index << " device" << std::endl;
                    return 1;
                }
            }
            Tensor normalized_state = qwen_rmsnorm_reference(
                hidden_state, layer.input_layernorm, layer_backend.get(),
                layer_device, config.rms_norm_eps);
            if (normalized_state.buffer() == nullptr) {
                std::cerr << "[ERROR] Layer " << layer_index
                          << " RMSNorm failed" << std::endl;
                return 1;
            }
            Tensor attention_output;
            if (layer.is_attention_layer) {
                attention_output = qwen_attention_reference(
                    normalized_state, layer, layer_backend.get(),
                    layer_device, config,
                    &attention_caches[static_cast<size_t>(layer_index)],
                    max_cache_len);
                if (attention_output.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " attention failed" << std::endl;
                    return 1;
                }
                hidden_state = make_residual(
                    hidden_state, attention_output,
                    layer_backend.get(), layer_device);
                if (hidden_state.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " attention residual failed" << std::endl;
                    return 1;
                }
                Tensor post_norm_state = qwen_rmsnorm_reference(
                    hidden_state, layer.post_attention_layernorm,
                    layer_backend.get(), layer_device, config.rms_norm_eps);
                if (post_norm_state.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " post-attention RMSNorm failed" << std::endl;
                    return 1;
                }
                normalized_state = std::move(post_norm_state);
            } else {
                attention_output = qwen_deltanet_reference(
                    normalized_state, layer, layer_backend.get(),
                    layer_device, config,
                    &deltanet_caches[static_cast<size_t>(layer_index)]);
                if (attention_output.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " DeltaNet failed" << std::endl;
                    return 1;
                }
                hidden_state = make_residual(
                    hidden_state, attention_output,
                    layer_backend.get(), layer_device);
                if (hidden_state.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " DeltaNet residual failed" << std::endl;
                    return 1;
                }
                Tensor post_norm_state = qwen_rmsnorm_reference(
                    hidden_state, layer.post_attention_layernorm,
                    layer_backend.get(), layer_device, config.rms_norm_eps);
                if (post_norm_state.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " post-DeltaNet RMSNorm failed" << std::endl;
                    return 1;
                }
                normalized_state = std::move(post_norm_state);
            }
            Tensor layer_mlp = qwen_mlp_reference(
                normalized_state, layer, layer_backend.get(), layer_device);
            if (layer_mlp.buffer() == nullptr) {
                std::cerr << "[ERROR] Layer " << layer_index
                          << " MLP failed" << std::endl;
                return 1;
            }
            hidden_state = make_residual(
                hidden_state, layer_mlp,
                layer_backend.get(), layer_device);
            if (hidden_state.buffer() == nullptr) {
                std::cerr << "[ERROR] Layer " << layer_index
                          << " residual failed" << std::endl;
                return 1;
            }
            if (verbose_step && ((layer_index + 1) % 8 == 0 ||
                layer_index + 1 == config.num_hidden_layers)) {
                std::cout << "  forward layers: "
                          << (layer_index + 1) << "/"
                          << config.num_hidden_layers << std::endl;
            }
        }

        const Device final_device = devices.back();
        auto final_backend =
            BackendRegistry::instance().get_backend(final_device);
        if (final_backend == nullptr || !final_backend->is_available()) {
            std::cerr << "[ERROR] Failed to create final device backend"
                      << std::endl;
            return 1;
        }
        if (hidden_state.device() != final_device) {
            hidden_state = hidden_state.to(final_device);
            if (hidden_state.buffer() == nullptr) {
                std::cerr << "[ERROR] Failed to transfer activation to final device"
                          << std::endl;
                return 1;
            }
        }
        Tensor final_hidden = qwen_rmsnorm_reference(
            hidden_state, weights.shared.final_norm, final_backend.get(),
            final_device, config.rms_norm_eps);
        if (final_hidden.buffer() == nullptr) {
            std::cerr << "[ERROR] Final norm failed" << std::endl;
            return 1;
        }

        Tensor logits = Linear::forward(
            final_hidden, weights.shared.lm_head, true);
        if (logits.buffer() == nullptr) {
            std::cerr << "[ERROR] Real lm_head failed" << std::endl;
            return 1;
        }
        status = final_backend->synchronize();
        if (!status.ok()) {
            std::cerr << "[ERROR] Final projection synchronization failed: "
                      << status.message() << std::endl;
            return 1;
        }
        int64_t next_id = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
        bool device_argmax = false;
        if (logits.device().type() == DeviceType::DiscreteGPU) {
            auto result_buffer = final_backend->create_buffer(
                sizeof(int64_t), MemoryType::Device);
            if (result_buffer != nullptr) {
                status = final_backend->argmax_last_row(
                    result_buffer.get(), logits.buffer().get(), logits.dtype(),
                    logits.shape().dim(0), logits.shape().dim(1));
                if (status.ok()) {
                    status = final_backend->memcpy_d2h(
                        &next_id, result_buffer.get(), sizeof(next_id));
                    device_argmax = status.ok();
                }
            }
        }
        if (!device_argmax) {
            std::vector<uint16_t> logits_host(
                static_cast<size_t>(logits.numel()));
            status = final_backend->memcpy_d2h(
                logits_host.data(), logits.buffer().get(), logits.nbytes());
            if (!status.ok()) {
                std::cerr << "[ERROR] Logits D2H failed: " << status.message()
                          << std::endl;
                return 1;
            }
            const size_t last_offset =
                (static_cast<size_t>(logits.shape().dim(0)) - 1) *
                static_cast<size_t>(logits.shape().dim(1));
            for (int64_t token = 0; token < logits.shape().dim(1); ++token) {
                uint32_t bits = static_cast<uint32_t>(logits_host[
                    last_offset + static_cast<size_t>(token)]) << 16;
                float value = 0.0f;
                std::memcpy(&value, &bits, sizeof(float));
                if (value > best_logit) {
                    best_logit = value;
                    next_id = token;
                }
            }
        }
        generated_ids.push_back(next_id);
        if (!is_prefill) ++decode_token_count;
        if (verbose_step) {
            std::cout << "Full GPU forward + lm_head ok: logits=["
                      << logits.shape().dim(0) << "," << logits.shape().dim(1)
                      << "]" << std::endl;
            std::cout << "[Greedy step " << (generation_step + 1)
                      << "] id=" << next_id;
            if (!device_argmax) std::cout << ", logit=" << best_logit;
            std::cout << (device_argmax ? " (device argmax)" : "")
                      << std::endl;
        }
        const bool is_eos = std::binary_search(
            eos_token_ids.begin(), eos_token_ids.end(), next_id);
        if (is_eos) {
            if (verbose_step) {
                std::cout << "[Stop] EOS token generated." << std::endl;
            }
            break;
        }
        }
        const double decode_seconds = decode_token_count > 0
            ? std::chrono::duration<double>(SteadyClock::now() - decode_start)
                  .count()
            : 0.0;
        const double decode_tps = decode_seconds > 0.0
            ? static_cast<double>(decode_token_count) / decode_seconds
            : 0.0;
        const std::vector<int64_t> generated_suffix(
            generated_ids.begin() + static_cast<ptrdiff_t>(prompt_token_count),
            generated_ids.end());
        std::cout << "\n[Generated text]\n"
                  << decode_ids(generated_suffix, true) << std::endl;
        if (decode_token_count > 0) {
            std::cout << "[Performance] decode_tokens=" << decode_token_count
                      << ", decode_seconds=" << decode_seconds
                      << ", decode_tps=" << decode_tps << std::endl;
        } else {
            std::cout << "[Performance] decode_tokens=0, decode_tps=N/A"
                      << std::endl;
        }
        return 0;
    }

    // 3. 选择设备并创建 backend（先于权重加载，确保 GPU 识别正确）
    Device device = select_device(backend_name);
    std::cout << "\n[Device] backend=" << device.backend()
              << ", id=" << device.id()
              << ", type=" << DeviceTypeToString(device.type())
              << ", unified=" << device.unified_memory_supported()
              << std::endl;

    auto backend = BackendRegistry::instance().get_backend(device);
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

    // 6. 按模型配置加载全部真实层。
    const int64_t k_probe_layers = config.num_hidden_layers;
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
        if (layer.q_norm.buffer() != nullptr) {
            std::cout << "  q_norm: shape=[" << layer.q_norm.shape().dim(0)
                      << "] dtype=" << static_cast<int>(layer.q_norm.dtype())
                      << std::endl;
        }
        if (layer.k_norm.buffer() != nullptr) {
            std::cout << "  k_norm: shape=[" << layer.k_norm.shape().dim(0)
                      << "] dtype=" << static_cast<int>(layer.k_norm.dtype())
                      << std::endl;
        }
        inspect_quantized_weight("mlp_gate_proj", layer.mlp_gate_proj);
        inspect_quantized_weight("mlp_up_proj", layer.mlp_up_proj);
        inspect_quantized_weight("mlp_down_proj", layer.mlp_down_proj);
        if (!layer.is_attention_layer) {
            inspect_tensor("linear_attn_norm", layer.linear_attn_norm);
            inspect_tensor("a_log", layer.a_log);
            inspect_tensor("conv1d_weight", layer.conv1d_weight);
            inspect_tensor("dt_bias", layer.dt_bias);
            inspect_tensor("in_proj_a", layer.in_proj_a);
            inspect_tensor("in_proj_b", layer.in_proj_b);
            inspect_quantized_weight("in_proj_qkv", layer.in_proj_qkv);
            inspect_quantized_weight("in_proj_z", layer.in_proj_z);
            inspect_quantized_weight("linear_out_proj", layer.linear_out_proj);
        }
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
    // 使用 tokenizer_config.json 中的 Qwen3.5 对话模板。裸文本提示会让
    // causal LM 继续补空格或普通文本，不能用于判断 assistant 输出是否正常。
    // 这里显式关闭 thinking，对应模板中的空 <think> 段。
    std::string prompt;
    if (tok_status.ok()) {
        prompt = tokenizer.build_chat_prompt(
            {{"user", "Hello, how are you? And who are you?"}}, true, false);
    } else {
        prompt =
            "<|im_start|>user\nHello, how are you? And who are you?<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n";
    }
    std::vector<int64_t> input_ids = tokenize(prompt);
    std::cout << "\n[Prompt -> Token IDs]" << std::endl;
    std::cout << "  prompt: " << prompt << std::endl;
    std::cout << "  ids:    [";
    for (size_t i = 0; i < input_ids.size(); ++i) {
        if (i) std::cout << ", ";
        std::cout << input_ids[i];
    }
    std::cout << "]" << std::endl;
    std::cout << "  decoded prompt: " << decode_ids(input_ids)
          << std::endl;

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

namespace {

Tensor qwen_rmsnorm_reference(const Tensor& input, const Tensor& weight,
                              Backend* backend, const Device& device,
                              float eps) {
    if (backend == nullptr || input.dtype() != weight.dtype() ||
        input.buffer() == nullptr ||
        weight.buffer() == nullptr || input.shape().ndim() == 0 ||
        weight.shape().ndim() != 1 ||
        weight.shape().dim(0) != input.shape().dim(input.shape().ndim() - 1) ||
        input.device() != device || weight.device() != device) {
        return Tensor();
    }
    return RMSNorm::forward(input, weight, eps, nullptr, true);
}

Tensor make_residual(const Tensor& lhs, const Tensor& rhs, Backend* backend,
                     const Device& device) {
    if (backend == nullptr || lhs.dtype() != rhs.dtype() ||
        lhs.shape() != rhs.shape() || lhs.device() != device ||
        rhs.device() != device || lhs.buffer() == nullptr ||
        rhs.buffer() == nullptr) {
        return Tensor();
    }
    const size_t count = static_cast<size_t>(lhs.numel());
    auto buffer = backend->create_buffer(lhs.nbytes(),
                                         device.is_gpu() ? MemoryType::Device
                                                         : MemoryType::Host);
    if (buffer == nullptr) return Tensor();
    if (device.is_gpu()) {
        Status status = backend->add(buffer.get(), lhs.buffer().get(),
                                     rhs.buffer().get(), lhs.dtype(),
                                     static_cast<int64_t>(count));
        if (!status.ok()) return Tensor();
    } else {
        if (lhs.dtype() != DType::FP32) return Tensor();
        const float* a = static_cast<const float*>(lhs.data());
        const float* b = static_cast<const float*>(rhs.data());
        float* out = static_cast<float*>(buffer->data());
        for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
    }
    return Tensor(lhs.shape(), lhs.dtype(), device, std::move(buffer));
}

Tensor qwen_mlp_reference(const Tensor& input, const Qwen3LayerWeights& layer,
                          Backend* backend, const Device& device) {
    if (backend == nullptr || input.buffer() == nullptr ||
        input.device() != device ||
        input.dtype() != layer.mlp_gate_proj.values.dtype()) {
        return Tensor();
    }
    Status validation = Linear::validate(input, layer.mlp_gate_proj.values, true,
                                         nullptr);
    if (!validation.ok()) {
        std::cerr << "[MLP] gate validation failed: " << validation.message()
                  << std::endl;
        return Tensor();
    }
    validation = Linear::validate(input, layer.mlp_up_proj.values, true, nullptr);
    if (!validation.ok()) {
        std::cerr << "[MLP] up validation failed: " << validation.message()
                  << std::endl;
        return Tensor();
    }
    Tensor gate = Linear::forward(input, layer.mlp_gate_proj.values, true);
    Tensor up = Linear::forward(input, layer.mlp_up_proj.values, true);
    if (gate.buffer() == nullptr || up.buffer() == nullptr) {
        std::cerr << "[MLP] gate/up GEMM returned an empty tensor; gate shape=["
                  << layer.mlp_gate_proj.values.shape().dim(0) << ","
                  << layer.mlp_gate_proj.values.shape().dim(1)
                  << "], dtype=" << static_cast<int>(layer.mlp_gate_proj.values.dtype())
                  << std::endl;
        return Tensor();
    }
    Tensor activated = Elementwise::silu_mul(gate, up);
    if (activated.buffer() == nullptr) return Tensor();
    validation = Linear::validate(activated, layer.mlp_down_proj.values, true,
                                  nullptr);
    if (!validation.ok()) {
        std::cerr << "[MLP] down validation failed: " << validation.message()
                  << std::endl;
        return Tensor();
    }
    Tensor output = Linear::forward(activated, layer.mlp_down_proj.values, true);
    if (output.buffer() == nullptr) {
        std::cerr << "[MLP] down GEMM returned an empty tensor; weight shape=["
                  << layer.mlp_down_proj.values.shape().dim(0) << ","
                  << layer.mlp_down_proj.values.shape().dim(1) << "]"
                  << std::endl;
    }
    return output;
}

Tensor qwen_attention_reference(const Tensor& input,
                                const Qwen3LayerWeights& layer,
                                Backend* backend, const Device& device,
                                const Qwen3Config& config,
                                AttentionKVCache* cache,
                                int64_t max_cache_len) {
    (void)device;
    if (!layer.is_attention_layer || backend == nullptr) return Tensor();
    const Tensor& wq = layer.q_proj.values;
    const Tensor& wk = layer.k_proj.values;
    const Tensor& wv = layer.v_proj.values;
    const Tensor& wo = layer.o_proj.values;
    auto describe = [](const Tensor& tensor) {
        if (tensor.buffer() == nullptr) return std::string("empty");
        return std::string("shape=[") +
               std::to_string(tensor.shape().dim(0)) + "," +
               std::to_string(tensor.shape().dim(1)) + "] dtype=" +
               std::to_string(static_cast<int>(tensor.dtype())) +
               " device=" + tensor.device().backend() + ":" +
               std::to_string(tensor.device().id());
    };
    Status validation = GatedGQAAttention::validate(
        input, wq, wk, wv, wo, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, config.rope_head_dim,
        config.rope_theta);
    if (!validation.ok()) {
        std::cerr << "[Attention] validation failed: " << validation.message()
                  << " input=" << describe(input) << " q=" << describe(wq)
                  << " k=" << describe(wk) << " v=" << describe(wv)
                  << " o=" << describe(wo) << std::endl;
        return Tensor();
    }
    if (cache != nullptr) {
        return GatedGQAAttention::forward_cached(
            input, wq, wk, wv, wo, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim,
            config.rope_head_dim, max_cache_len, cache, config.rope_theta,
            nullptr, layer.q_norm, layer.k_norm, config.rms_norm_eps);
    }
    return GatedGQAAttention::forward(
        input, wq, wk, wv, wo, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, config.rope_head_dim,
        config.rope_theta, nullptr, layer.q_norm, layer.k_norm,
        config.rms_norm_eps);
}

Tensor qwen_deltanet_reference(const Tensor& input,
                               const Qwen3LayerWeights& layer,
                               Backend* backend, const Device& device,
                               const Qwen3Config& config,
                               DeltaNetCache* cache) {
    DeltaNetCache local_cache;
    DeltaNetCache* active_cache = cache == nullptr ? &local_cache : cache;
    if (backend == nullptr || input.buffer() == nullptr ||
        !input.device().is_gpu() || input.device() != device ||
        input.dtype() != DType::BF16 || input.shape().ndim() != 2 ||
        layer.in_proj_qkv.values.buffer() == nullptr ||
        layer.in_proj_z.values.buffer() == nullptr ||
        layer.linear_out_proj.values.buffer() == nullptr) return Tensor();
    const int64_t qk_heads = config.linear_num_key_heads;
    const int64_t value_heads = config.linear_num_value_heads;
    const int64_t key_dim = config.linear_key_head_dim;
    const int64_t value_dim = config.linear_value_head_dim;
    const int64_t tokens = input.shape().dim(0);
    Tensor qkv_projection = Linear::forward(
        input, layer.in_proj_qkv.values, true);
    Tensor z = Linear::forward(input, layer.in_proj_z.values, true);
    Tensor a = Linear::forward(input, layer.in_proj_a, true);
    Tensor b = Linear::forward(input, layer.in_proj_b, true);
    if (qkv_projection.buffer() == nullptr || z.buffer() == nullptr ||
        a.buffer() == nullptr || b.buffer() == nullptr) return Tensor();
    DeltaNetQKV qkv = GatedDeltaNet::grouped_causal_conv(
        qkv_projection, layer.conv1d_weight, qk_heads, value_heads, key_dim,
        value_dim, config.linear_conv_kernel_dim, active_cache);
    if (!qkv.valid()) return Tensor();
    Tensor recurrent = GatedDeltaNet::recurrent(
        qkv.query, qkv.key, qkv.value,
        a.reshape(Shape{tokens, value_heads}),
        b.reshape(Shape{tokens, value_heads}),
        z.reshape(Shape{tokens, value_heads, value_dim}), layer.a_log,
        layer.dt_bias, layer.linear_attn_norm, qk_heads, value_heads, key_dim,
        value_dim, config.rms_norm_eps, active_cache);
    if (recurrent.buffer() == nullptr) return Tensor();
    Tensor projected = Linear::forward(
        recurrent.reshape(Shape{tokens, value_heads * value_dim}),
        layer.linear_out_proj.values, true);
    return projected;
}

} // namespace