// qwen_infer.cc
// Qwen3.5/Qwen3.8 27B BF16 多卡加载与推理入口。
//
// 当前 HIP 路径已将完整文本权重按连续层加载到最多八张离散 GPU；后续
// 在此入口接入 BF16 kernel、partition activation 传输和 prefill/decode。
//
// 用法：qwen_infer <model_dir> [backend=cpu|hip] [max_devices=8]

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
#include "tokenizer/qwen_tokenizer.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"
#include "ops/softmax.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <memory>
#include <limits>
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
              << " <model_dir> [backend=cpu|hip] [max_devices=8]"
              << std::endl;
    std::cout << "Example:\n  " << program
              << " /path/to/Qwen3.8-27B hip 8" << std::endl;
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

    const size_t embedding_bytes =
        static_cast<size_t>(embedding.shape().numel()) *
        SizeOfDType(DType::BF16);
    std::vector<uint16_t> host_embedding(
        static_cast<size_t>(embedding.shape().numel()));
    Status status = backend->memcpy_d2h(
        host_embedding.data(), embedding.buffer().get(), embedding_bytes);
    if (!status.ok()) return status;

    std::vector<uint16_t> host_output(
        ids.size() * static_cast<size_t>(hidden_size));
    for (size_t row = 0; row < ids.size(); ++row) {
        const size_t source_offset =
            static_cast<size_t>(ids[row]) *
            static_cast<size_t>(hidden_size);
        std::memcpy(host_output.data() +
                        row * static_cast<size_t>(hidden_size),
                    host_embedding.data() + source_offset, row_bytes);
    }

    status = backend->memcpy_h2d(buffer.get(), host_output.data(),
                                 output_bytes);
    if (!status.ok()) return status;

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

Tensor bf16_to_fp32(const Tensor& input, Backend* backend,
                    const Device& device);
Tensor fp32_to_bf16(const Tensor& input, Backend* backend,
                    const Device& device);
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
                                const Qwen3Config& config);
Tensor qwen_deltanet_reference(const Tensor& input,
                               const Qwen3LayerWeights& layer,
                               Backend* backend, const Device& device,
                               const Qwen3Config& config);

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
    int max_devices = (argc > 3) ? std::atoi(argv[3]) : 8;
    if (max_devices <= 0) max_devices = 1;
    if (max_devices > 8) max_devices = 8;

    std::cout << "========================================" << std::endl;
    std::cout << "Qwen3.5/Qwen3.8 BF16 Multi-GPU Loader" << std::endl;
    std::cout << "Model dir: " << model_dir << std::endl;
    std::cout << "Backend:   " << backend_name << std::endl;
    std::cout << "Max devices: " << max_devices << std::endl;
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
            BackendRegistry::instance().create_backend(devices.back());
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
            BackendRegistry::instance().create_backend(devices.front());
        if (embedding_backend == nullptr ||
            !embedding_backend->is_available()) {
            std::cerr << "[ERROR] Failed to create backend for embedding"
                      << std::endl;
            return 1;
        }
        constexpr int kMaxNewTokens = 8;
        std::vector<int64_t> generated_ids = input_ids;
        for (int generation_step = 0; generation_step < kMaxNewTokens;
             ++generation_step) {
            input_ids = generated_ids;
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

        const size_t embedded_bytes =
            static_cast<size_t>(embedded.shape().numel()) *
            SizeOfDType(DType::BF16);
        std::vector<uint16_t> embedded_host(
            static_cast<size_t>(embedded.shape().numel()));
        status = embedding_backend->memcpy_d2h(
            embedded_host.data(), embedded.buffer().get(), embedded_bytes);
        if (!status.ok()) {
            std::cerr << "[ERROR] Embedding D2H check failed: "
                      << status.message() << std::endl;
            return 1;
        }
        std::cout << "Real BF16 embedding lookup ok: ["
                  << embedded.shape().dim(0) << ","
                  << embedded.shape().dim(1) << "], D2H first value=0x"
                  << std::hex << embedded_host.front() << std::dec
                  << std::endl;

        // 首段真实前向：embedding -> input RMSNorm -> MLP -> residual。
        // 目前 RMSNorm/激活仍是 FP32 reference，因此这里显式做 BF16/FP32
        // staging；权重本身仍直接使用 GPU 上的真实 BF16 tensor。
        const Qwen3LayerWeights& first_layer = weights.layers.front();
        Tensor hidden_fp32 = bf16_to_fp32(
            embedded, embedding_backend.get(), devices.front());
        Tensor norm_input = bf16_to_fp32(
            first_layer.input_layernorm, embedding_backend.get(), devices.front());
        if (hidden_fp32.buffer() == nullptr || norm_input.buffer() == nullptr) {
            std::cerr << "[ERROR] Failed to stage embedding/layer norm to FP32"
                      << std::endl;
            return 1;
        }
        Tensor normalized = qwen_rmsnorm_reference(
            hidden_fp32, norm_input, embedding_backend.get(), devices.front());
        if (normalized.buffer() == nullptr) {
            std::cerr << "[ERROR] Real layer RMSNorm failed" << std::endl;
            return 1;
        }
        Tensor linear_output;
        if (first_layer.is_attention_layer) {
            linear_output = qwen_attention_reference(
                normalized, first_layer, embedding_backend.get(), devices.front(),
                config);
        } else {
            linear_output = qwen_deltanet_reference(
                normalized, first_layer, embedding_backend.get(), devices.front(),
                config);
        }
        if (linear_output.buffer() == nullptr) {
            std::cerr << "[ERROR] Real layer attention/DeltaNet failed" << std::endl;
            return 1;
        }
        Tensor layer0_output = make_residual(
            hidden_fp32, linear_output, embedding_backend.get(),
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
        std::vector<float> layer0_host(static_cast<size_t>(layer0_output.numel()));
        status = embedding_backend->memcpy_d2h(
            layer0_host.data(), layer0_output.buffer().get(), layer0_output.nbytes());
        if (!status.ok()) {
            std::cerr << "[ERROR] Real layer D2H check failed: "
                      << status.message() << std::endl;
            return 1;
        }
        std::cout << "Real layer 0 attention/DeltaNet path ok: output=["
                  << layer0_output.shape().dim(0) << ","
                  << layer0_output.shape().dim(1) << "], first="
                  << layer0_host.front() << std::endl;

        // 补齐第 0 层的 post-mixer RMSNorm、MLP 和第二个 residual。
        // Decoder layer 的两个子层都必须执行，不能只验证第一个 mixer。
        Tensor layer0_post_norm_weight = bf16_to_fp32(
            first_layer.post_attention_layernorm, embedding_backend.get(),
            devices.front());
        Tensor layer0_post_norm = qwen_rmsnorm_reference(
            layer0_output, layer0_post_norm_weight, embedding_backend.get(),
            devices.front(), config.rms_norm_eps);
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
        Tensor layer0_mlp_fp32 = bf16_to_fp32(
            layer0_mlp, embedding_backend.get(), devices.front());
        Tensor hidden_state = make_residual(
            layer0_output, layer0_mlp_fp32, embedding_backend.get(),
            devices.front());
        if (hidden_state.buffer() == nullptr) {
            std::cerr << "[ERROR] Layer 0 MLP residual failed" << std::endl;
            return 1;
        }
        print_tensor_stats("layer0.output", layer0_output,
                           embedding_backend.get());
        print_tensor_stats("layer0.mlp", layer0_mlp_fp32,
                           embedding_backend.get());
        print_tensor_stats("layer0.hidden", hidden_state,
                           embedding_backend.get());

        // 继续执行其余层的真实 Attention/DeltaNet、MLP 和残差路径。
        // 这是便于数值对齐的单卡 reference path；尚未使用 KV/state cache。
        for (int64_t layer_index = 1;
             layer_index < config.num_hidden_layers; ++layer_index) {
            const Qwen3LayerWeights& layer =
                weights.layers[static_cast<size_t>(layer_index)];
            const Device layer_device =
                weights.layer_devices[static_cast<size_t>(layer_index)];
            auto layer_backend =
                BackendRegistry::instance().create_backend(layer_device);
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
            Tensor layer_norm = bf16_to_fp32(
                layer.input_layernorm, layer_backend.get(), layer_device);
            if (layer_norm.buffer() == nullptr) {
                std::cerr << "[ERROR] Failed to stage layer " << layer_index
                          << " input norm" << std::endl;
                return 1;
            }
            Tensor normalized_state = qwen_rmsnorm_reference(
                hidden_state, layer_norm, layer_backend.get(),
                layer_device);
            if (normalized_state.buffer() == nullptr) {
                std::cerr << "[ERROR] Layer " << layer_index
                          << " RMSNorm failed" << std::endl;
                return 1;
            }
            Tensor attention_output;
            if (layer.is_attention_layer) {
                attention_output = qwen_attention_reference(
                    normalized_state, layer, layer_backend.get(),
                    layer_device, config);
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
                Tensor post_norm_weight = bf16_to_fp32(
                    layer.post_attention_layernorm, layer_backend.get(),
                    layer_device);
                Tensor post_norm_state = qwen_rmsnorm_reference(
                    hidden_state, post_norm_weight, layer_backend.get(),
                    layer_device);
                if (post_norm_state.buffer() == nullptr) {
                    std::cerr << "[ERROR] Layer " << layer_index
                              << " post-attention RMSNorm failed" << std::endl;
                    return 1;
                }
                normalized_state = std::move(post_norm_state);
            } else {
                attention_output = qwen_deltanet_reference(
                    normalized_state, layer, layer_backend.get(),
                    layer_device, config);
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
                Tensor post_norm_weight = bf16_to_fp32(
                    layer.post_attention_layernorm, layer_backend.get(),
                    layer_device);
                Tensor post_norm_state = qwen_rmsnorm_reference(
                    hidden_state, post_norm_weight, layer_backend.get(),
                    layer_device);
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
            Tensor layer_mlp_fp32 = bf16_to_fp32(
                layer_mlp, layer_backend.get(), layer_device);
            hidden_state = make_residual(
                hidden_state, layer_mlp_fp32,
                layer_backend.get(), layer_device);
            if (hidden_state.buffer() == nullptr) {
                std::cerr << "[ERROR] Layer " << layer_index
                          << " residual failed" << std::endl;
                return 1;
            }
            if (layer_index == 1 || layer_index == 3 || layer_index == 4 ||
                layer_index == 63) {
                print_tensor_stats("layer.hidden", hidden_state,
                                   layer_backend.get());
            }
            if ((layer_index + 1) % 8 == 0 ||
                layer_index + 1 == config.num_hidden_layers) {
                std::cout << "  reference forward layers: "
                          << (layer_index + 1) << "/"
                          << config.num_hidden_layers << std::endl;
            }
        }

        const Device final_device = devices.back();
        auto final_backend =
            BackendRegistry::instance().create_backend(final_device);
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
        Tensor final_norm_weight = bf16_to_fp32(
            weights.shared.final_norm, final_backend.get(), final_device);
        if (final_norm_weight.buffer() == nullptr) {
            std::cerr << "[ERROR] Failed to stage final norm" << std::endl;
            return 1;
        }
        Tensor final_hidden = qwen_rmsnorm_reference(
            hidden_state, final_norm_weight, final_backend.get(), final_device);
        Tensor final_hidden_bf16 = fp32_to_bf16(
            final_hidden, final_backend.get(), final_device);
        if (final_hidden_bf16.buffer() == nullptr) {
            std::cerr << "[ERROR] Final norm failed" << std::endl;
            return 1;
        }

        Tensor logits = Linear::forward(
            final_hidden_bf16, weights.shared.lm_head, true);
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
        std::vector<uint16_t> logits_host(static_cast<size_t>(logits.numel()));
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
        int64_t next_id = 0;
        float best_logit = -std::numeric_limits<float>::infinity();
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
        generated_ids.push_back(next_id);
        std::cout << "Full reference forward + lm_head ok: logits=["
                  << logits.shape().dim(0) << "," << logits.shape().dim(1)
                  << "]" << std::endl;
        std::cout << "[Greedy step " << (generation_step + 1)
              << "] id=" << next_id
                  << ", logit=" << best_logit << std::endl;
        }
        std::cout << "\n[Generated text]\n" << decode_ids(generated_ids)
              << std::endl;
        return 0;
    }

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
            {{"user", "Hello, how are you?"}}, true);
        prompt += "<think>\n\n</think>\n\n";
    } else {
        prompt =
            "<|im_start|>user\nHello, how are you?<|im_end|>\n"
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

// 当前 HIP 算子仍以 FP32 host/unified tensor 为输入。这个适配器只用于
// 单卡功能验证：把设备上的 BF16 tensor 复制到主机，转换为 FP32 后再上传。
Tensor bf16_to_fp32(const Tensor& input, Backend* backend, const Device& device) {
    if (backend == nullptr || input.dtype() != DType::BF16 ||
        input.buffer() == nullptr) {
        return Tensor();
    }
    std::vector<uint16_t> src(static_cast<size_t>(input.numel()));
    if (!backend->memcpy_d2h(src.data(), input.buffer().get(), input.nbytes()).ok()) {
        return Tensor();
    }
    std::vector<float> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
        std::memcpy(&dst[i], &bits, sizeof(float));
    }
    auto buffer = backend->create_buffer(dst.size() * sizeof(float), MemoryType::Unified);
    if (buffer == nullptr || !backend->memcpy_h2d(buffer.get(), dst.data(),
                                                   dst.size() * sizeof(float)).ok()) {
        return Tensor();
    }
    return Tensor(input.shape(), DType::FP32, device, std::move(buffer));
}

Tensor fp32_to_bf16(const Tensor& input, Backend* backend,
                    const Device& device) {
    if (backend == nullptr || input.dtype() != DType::FP32 ||
        input.buffer() == nullptr) {
        return Tensor();
    }
    std::vector<float> src(static_cast<size_t>(input.numel()));
    if (!backend->memcpy_d2h(src.data(), input.buffer().get(), input.nbytes()).ok()) {
        return Tensor();
    }
    std::vector<uint16_t> dst(src.size());
    for (size_t i = 0; i < src.size(); ++i) {
        uint32_t bits = 0;
        std::memcpy(&bits, &src[i], sizeof(float));
        dst[i] = static_cast<uint16_t>(bits >> 16);
    }
    auto buffer = backend->create_buffer(dst.size() * sizeof(uint16_t), MemoryType::Device);
    if (buffer == nullptr || !backend->memcpy_h2d(buffer.get(), dst.data(),
                                                   dst.size() * sizeof(uint16_t)).ok()) {
        return Tensor();
    }
    return Tensor(input.shape(), DType::BF16, device, std::move(buffer));
}

Tensor qwen_rmsnorm_reference(const Tensor& input, const Tensor& weight,
                              Backend* backend, const Device& device,
                              float eps) {
    if (backend == nullptr || input.dtype() != DType::FP32 ||
        weight.dtype() != DType::FP32 || input.buffer() == nullptr ||
        weight.buffer() == nullptr || input.shape().ndim() == 0 ||
        weight.shape().ndim() != 1 ||
        weight.shape().dim(0) != input.shape().dim(input.shape().ndim() - 1)) {
        return Tensor();
    }
    const int64_t hidden_size = input.shape().dim(input.shape().ndim() - 1);
    const int64_t rows = input.numel() / hidden_size;
    std::vector<float> x(static_cast<size_t>(input.numel()));
    std::vector<float> w(static_cast<size_t>(weight.numel()));
    if (!backend->memcpy_d2h(x.data(), input.buffer().get(), input.nbytes()).ok() ||
        !backend->memcpy_d2h(w.data(), weight.buffer().get(), weight.nbytes()).ok()) {
        return Tensor();
    }
    for (int64_t row = 0; row < rows; ++row) {
        float mean_sq = 0.0f;
        float* values = x.data() + static_cast<size_t>(row * hidden_size);
        for (int64_t d = 0; d < hidden_size; ++d) mean_sq += values[d] * values[d];
        const float inv_rms = 1.0f / std::sqrt(mean_sq / hidden_size + eps);
        for (int64_t d = 0; d < hidden_size; ++d) {
            values[d] = values[d] * inv_rms * (1.0f + w[static_cast<size_t>(d)]);
        }
    }
    auto buffer = backend->create_buffer(x.size() * sizeof(float), MemoryType::Unified);
    if (buffer == nullptr ||
        !backend->memcpy_h2d(buffer.get(), x.data(), x.size() * sizeof(float)).ok()) {
        return Tensor();
    }
    return Tensor(input.shape(), DType::FP32, device, std::move(buffer));
}

Tensor make_residual(const Tensor& lhs, const Tensor& rhs, Backend* backend,
                     const Device& device) {
    if (backend == nullptr || lhs.dtype() != DType::FP32 ||
        rhs.dtype() != DType::FP32 || lhs.shape() != rhs.shape()) {
        return Tensor();
    }
    const size_t count = static_cast<size_t>(lhs.numel());
    std::vector<float> a(count), b(count), out(count);
    if (!backend->memcpy_d2h(a.data(), lhs.buffer().get(), lhs.nbytes()).ok() ||
        !backend->memcpy_d2h(b.data(), rhs.buffer().get(), rhs.nbytes()).ok()) {
        return Tensor();
    }
    for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
    auto buffer = backend->create_buffer(count * sizeof(float), MemoryType::Unified);
    if (buffer == nullptr || !backend->memcpy_h2d(buffer.get(), out.data(),
                                                   out.size() * sizeof(float)).ok()) {
        return Tensor();
    }
    return Tensor(lhs.shape(), DType::FP32, device, std::move(buffer));
}

Tensor qwen_mlp_reference(const Tensor& input, const Qwen3LayerWeights& layer,
                          Backend* backend, const Device& device) {
    Tensor input_bf16 = fp32_to_bf16(input, backend, device);
    if (input_bf16.buffer() == nullptr) {
        std::cerr << "[MLP] FP32 -> BF16 activation conversion failed"
                  << std::endl;
        return Tensor();
    }
    Status validation = Linear::validate(input_bf16, layer.mlp_gate_proj.values, true,
                                         nullptr);
    if (!validation.ok()) {
        std::cerr << "[MLP] gate validation failed: " << validation.message()
                  << std::endl;
        return Tensor();
    }
    validation = Linear::validate(input_bf16, layer.mlp_up_proj.values, true, nullptr);
    if (!validation.ok()) {
        std::cerr << "[MLP] up validation failed: " << validation.message()
                  << std::endl;
        return Tensor();
    }
    Tensor gate = Linear::forward(input_bf16, layer.mlp_gate_proj.values, true);
    Tensor up = Linear::forward(input_bf16, layer.mlp_up_proj.values, true);
    if (gate.buffer() == nullptr || up.buffer() == nullptr) {
        std::cerr << "[MLP] gate/up GEMM returned an empty tensor; gate shape=["
                  << layer.mlp_gate_proj.values.shape().dim(0) << ","
                  << layer.mlp_gate_proj.values.shape().dim(1)
                  << "], dtype=" << static_cast<int>(layer.mlp_gate_proj.values.dtype())
                  << std::endl;
        return Tensor();
    }
    std::vector<float> gate_host(static_cast<size_t>(gate.numel()));
    std::vector<float> up_host(static_cast<size_t>(up.numel()));
    if (!backend->memcpy_d2h(gate_host.data(), gate.buffer().get(), gate.nbytes()).ok() ||
        !backend->memcpy_d2h(up_host.data(), up.buffer().get(), up.nbytes()).ok()) {
        return Tensor();
    }
    for (size_t i = 0; i < gate_host.size(); ++i) {
        const float x = gate_host[i];
        gate_host[i] = x / (1.0f + std::exp(-x)) * up_host[i];
    }
    auto act_fp32 = backend->create_buffer(gate_host.size() * sizeof(float), MemoryType::Unified);
    if (act_fp32 == nullptr || !backend->memcpy_h2d(act_fp32.get(), gate_host.data(),
                                                     gate_host.size() * sizeof(float)).ok()) {
        return Tensor();
    }
    Tensor activated_fp32(gate.shape(), DType::FP32, device, std::move(act_fp32));
    Tensor activated = fp32_to_bf16(activated_fp32, backend, device);
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
                                const Qwen3Config& config) {
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
    return GatedGQAAttention::forward(
        input, wq, wk, wv, wo, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, config.rope_head_dim,
        config.rope_theta, nullptr, layer.q_norm, layer.k_norm,
        config.rms_norm_eps);
}

Tensor qwen_deltanet_reference(const Tensor& input,
                               const Qwen3LayerWeights& layer,
                               Backend* backend, const Device& device,
                               const Qwen3Config& config) {
    if (backend == nullptr || input.buffer() == nullptr ||
        input.dtype() != DType::FP32 || layer.in_proj_qkv.values.buffer() == nullptr ||
        layer.in_proj_z.values.buffer() == nullptr ||
        layer.linear_out_proj.values.buffer() == nullptr) {
        return Tensor();
    }
    const int64_t qk_heads = config.linear_num_key_heads;
    const int64_t v_heads = config.linear_num_value_heads;
    const int64_t key_head_dim = config.linear_key_head_dim;
    const int64_t value_head_dim = config.linear_value_head_dim;
    const int64_t key_width = qk_heads * key_head_dim;
    const int64_t v_width = v_heads * value_head_dim;
    const int64_t seq_len = input.shape().dim(0);
    if (input.shape().ndim() != 2 || input.shape().dim(1) != 5120 ||
        layer.in_proj_qkv.values.shape().dim(0) != key_width * 2 + v_width ||
        layer.in_proj_z.values.shape().dim(0) != v_width) {
        return Tensor();
    }

    Tensor input_bf16 = fp32_to_bf16(input, backend, device);
    if (input_bf16.buffer() == nullptr) return Tensor();
    Tensor qkv_tensor = Linear::forward(input_bf16, layer.in_proj_qkv.values, true);
    Tensor z = Linear::forward(input_bf16, layer.in_proj_z.values, true);
    Tensor a_tensor = Linear::forward(input_bf16, layer.in_proj_a, true);
    Tensor b_tensor = Linear::forward(input_bf16, layer.in_proj_b, true);
    if (qkv_tensor.buffer() == nullptr || z.buffer() == nullptr ||
        a_tensor.buffer() == nullptr || b_tensor.buffer() == nullptr) return Tensor();

    std::vector<uint16_t> qkv_bf16(static_cast<size_t>(qkv_tensor.numel()));
    std::vector<uint16_t> z_bf16(static_cast<size_t>(z.numel()));
    if (!backend->memcpy_d2h(qkv_bf16.data(), qkv_tensor.buffer().get(), qkv_tensor.nbytes()).ok() ||
        !backend->memcpy_d2h(z_bf16.data(), z.buffer().get(), z.nbytes()).ok()) {
        return Tensor();
    }
    auto bf16_value = [](uint16_t x) {
        uint32_t bits = static_cast<uint32_t>(x) << 16;
        float value = 0.0f;
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    };
    std::vector<float> qkv_values(static_cast<size_t>(qkv_tensor.numel()));
    std::vector<float> zf(static_cast<size_t>(z.numel()));
    for (size_t i = 0; i < qkv_values.size(); ++i) qkv_values[i] = bf16_value(qkv_bf16[i]);
    for (size_t i = 0; i < zf.size(); ++i) zf[i] = bf16_value(z_bf16[i]);

    // Transformers first views the projection as
    // [num_key_heads, key_dim + key_dim + heads_per_group * value_dim]
    // and only then splits q/k/v.  Therefore the checkpoint rows are grouped
    // per key head, rather than stored as [all_q, all_k, all_v].  Rebuild the
    // flat stream consumed by the causal depthwise convolution.
    const int64_t heads_per_group = v_heads / qk_heads;
    const int64_t group_value_width = heads_per_group * value_head_dim;
    const int64_t group_width = key_head_dim * 2 + group_value_width;
    const int64_t qkv_width = key_width * 2 + v_width;
    std::vector<float> grouped_qkv = std::move(qkv_values);
    qkv_values.assign(static_cast<size_t>(seq_len * qkv_width), 0.0f);
    std::vector<float> grouped_z = std::move(zf);
    zf.assign(static_cast<size_t>(seq_len * v_width), 0.0f);
    for (int64_t t = 0; t < seq_len; ++t) {
        const float* source = grouped_qkv.data() + t * qkv_width;
        float* target = qkv_values.data() + t * qkv_width;
        for (int64_t group = 0; group < qk_heads; ++group) {
            const int64_t source_base = group * group_width;
            std::memcpy(target + group * key_head_dim,
                        source + source_base,
                        static_cast<size_t>(key_head_dim) * sizeof(float));
            std::memcpy(target + key_width + group * key_head_dim,
                        source + source_base + key_head_dim,
                        static_cast<size_t>(key_head_dim) * sizeof(float));
            std::memcpy(target + key_width * 2 +
                            group * group_value_width,
                        source + source_base + key_head_dim * 2,
                        static_cast<size_t>(group_value_width) * sizeof(float));
            std::memcpy(zf.data() + t * v_width + group * group_value_width,
                        grouped_z.data() + t * v_width +
                            group * group_value_width,
                        static_cast<size_t>(group_value_width) * sizeof(float));
        }
    }

    // Apply the causal depthwise conv1d used by Qwen's DeltaNet input path.
    // The weight is [10240, 1, 4]; zero-padding is used for the prefill prefix.
    std::vector<float> conv(qkv_values.size(), 0.0f);
    std::vector<uint16_t> conv_w_bf16(static_cast<size_t>(layer.conv1d_weight.numel()));
    if (!backend->memcpy_d2h(conv_w_bf16.data(), layer.conv1d_weight.buffer().get(),
                             layer.conv1d_weight.nbytes()).ok()) return Tensor();
    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t c = 0; c < qkv_width; ++c) {
            float value = 0.0f;
            for (int64_t j = 0; j < 4; ++j) {
                const int64_t source_t = t - 3 + j;
                if (source_t >= 0 && source_t < seq_len) {
                    value += qkv_values[static_cast<size_t>(source_t * qkv_width + c)] *
                             bf16_value(conv_w_bf16[static_cast<size_t>(c * 4 + j)]);
                }
            }
            // Qwen's causal depthwise convolution is followed by SiLU before
            // splitting the stream into q, k and v.
            conv[static_cast<size_t>(t * qkv_width + c)] =
                value / (1.0f + std::exp(-value));
        }
    }

    std::vector<uint16_t> a_bf16(static_cast<size_t>(a_tensor.numel()));
    std::vector<uint16_t> b_bf16(static_cast<size_t>(b_tensor.numel()));
    std::vector<uint16_t> alog_bf16(static_cast<size_t>(layer.a_log.numel()));
    std::vector<uint16_t> dt_bf16(static_cast<size_t>(layer.dt_bias.numel()));
    std::vector<uint16_t> norm_bf16(static_cast<size_t>(layer.linear_attn_norm.numel()));
    if (!backend->memcpy_d2h(a_bf16.data(), a_tensor.buffer().get(), a_tensor.nbytes()).ok() ||
        !backend->memcpy_d2h(b_bf16.data(), b_tensor.buffer().get(), b_tensor.nbytes()).ok() ||
        !backend->memcpy_d2h(alog_bf16.data(), layer.a_log.buffer().get(), layer.a_log.nbytes()).ok() ||
        !backend->memcpy_d2h(dt_bf16.data(), layer.dt_bias.buffer().get(), layer.dt_bias.nbytes()).ok() ||
        !backend->memcpy_d2h(norm_bf16.data(), layer.linear_attn_norm.buffer().get(), layer.linear_attn_norm.nbytes()).ok()) return Tensor();
    std::vector<float> a(static_cast<size_t>(seq_len * v_heads), 0.0f);
    std::vector<float> beta(a.size(), 0.0f);
    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t h = 0; h < v_heads; ++h) {
            const int64_t group = h / heads_per_group;
            const int64_t within = h % heads_per_group;
            const size_t grouped_index = static_cast<size_t>(
                t * v_heads + group * heads_per_group + within);
            const float av = bf16_value(a_bf16[grouped_index]);
            const float bv = bf16_value(b_bf16[grouped_index]);
            a[static_cast<size_t>(t * v_heads + h)] = av;
            beta[static_cast<size_t>(t * v_heads + h)] =
                1.0f / (1.0f + std::exp(-bv));
        }
    }

    std::vector<float> output(static_cast<size_t>(seq_len * v_width), 0.0f);
    std::vector<float> state(static_cast<size_t>(v_heads * key_head_dim * value_head_dim), 0.0f);
    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t vh = 0; vh < v_heads; ++vh) {
            const int64_t qh = vh / 3;
            // The recurrent decay is parameterized as
            // exp(-exp(A_log) * softplus(a + dt_bias)).
            const float decay_arg = bf16_value(dt_bf16[vh]) +
                                    a[static_cast<size_t>(t * v_heads + vh)];
            const float softplus_arg = std::max(decay_arg, 0.0f) +
                                       std::log1p(std::exp(-std::abs(decay_arg)));
            const float decay = std::exp(
                -std::exp(bf16_value(alog_bf16[vh])) * softplus_arg);
            float* s = state.data() + static_cast<size_t>(vh * key_head_dim * value_head_dim);
            for (int64_t i = 0; i < key_head_dim * value_head_dim; ++i) s[i] *= decay;
            const float* k = conv.data() + static_cast<size_t>(t * qkv_width + key_width + qh * key_head_dim);
            const float* v = conv.data() + static_cast<size_t>(t * qkv_width + key_width * 2 + vh * value_head_dim);
            const float b = beta[static_cast<size_t>(t * v_heads + vh)];
            float* out = output.data() + static_cast<size_t>(t * v_width + vh * value_head_dim);
            float q_norm = 0.0f;
            float k_norm = 0.0f;
            const float* q = conv.data() + static_cast<size_t>(t * qkv_width + qh * key_head_dim);
            for (int64_t i = 0; i < key_head_dim; ++i) {
                q_norm += q[i] * q[i];
                k_norm += k[i] * k[i];
            }
            // Match transformers' FLA l2norm exactly: normalization is over
            // the vector sum, not the RMS mean.
            q_norm = 1.0f / std::sqrt(q_norm + 1e-6f);
            k_norm = 1.0f / std::sqrt(k_norm + 1e-6f);
            for (int64_t j = 0; j < value_head_dim; ++j) {
                float retrieved = 0.0f;
                for (int64_t i = 0; i < key_head_dim; ++i)
                    retrieved += (k[i] * k_norm) * s[i * value_head_dim + j];
                const float corrected = v[j] - b * retrieved;
                for (int64_t i = 0; i < key_head_dim; ++i)
                    s[i * value_head_dim + j] += b * (k[i] * k_norm) * corrected;
            }
            constexpr float query_scale = 1.0f / 11.313708498984761f;
            for (int64_t j = 0; j < value_head_dim; ++j)
                for (int64_t i = 0; i < key_head_dim; ++i)
                    out[j] += (q[i] * q_norm * query_scale) * s[i * value_head_dim + j];
            float rms = 0.0f;
            for (int64_t j = 0; j < value_head_dim; ++j) rms += out[j] * out[j];
            rms = 1.0f / std::sqrt(rms / value_head_dim + config.rms_norm_eps);
            for (int64_t j = 0; j < value_head_dim; ++j) {
                // DeltaNet uses RMSNormGated, whose learned weight is the
                // ordinary multiplicative weight (unlike Qwen3.5's main
                // decoder RMSNorm, which uses the (1 + weight) form).
                out[j] *= rms * bf16_value(norm_bf16[static_cast<size_t>(j)]);
            }
        }
    }
    // z is the SiLU gate for the value stream.
    for (size_t i = 0; i < output.size(); ++i) {
        const float gate = zf[i];
        output[i] *= gate / (1.0f + std::exp(-gate));
    }
    auto out_buf = backend->create_buffer(output.size() * sizeof(float), MemoryType::Unified);
    if (out_buf == nullptr || !backend->memcpy_h2d(out_buf.get(), output.data(),
                                                    output.size() * sizeof(float)).ok()) return Tensor();
    Tensor recurrent(Shape{seq_len, v_width}, DType::FP32, device,
                       std::move(out_buf));
    Tensor recurrent_bf16 = fp32_to_bf16(
        recurrent, backend, device);
    if (recurrent_bf16.buffer() == nullptr) return Tensor();
    Tensor projected = Linear::forward(
        recurrent_bf16, layer.linear_out_proj.values, true);
    return bf16_to_fp32(projected, backend, device);
}

} // namespace