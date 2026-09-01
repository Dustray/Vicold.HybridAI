#define HYBRIDAI_EMBEDDED_QWEN_INFER
#include "../demo/qwen_infer.cc"

#include "hybrid.h"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <utility>

namespace hybridai {

ApiStatus::ApiStatus(ApiStatusCode code, std::string message)
    : code_(code), message_(std::move(message)) {}

bool ApiStatus::ok() const noexcept { return code_ == ApiStatusCode::OK; }
ApiStatusCode ApiStatus::code() const noexcept { return code_; }
const std::string& ApiStatus::message() const noexcept { return message_; }

class Generator::Impl {
public:
    explicit Impl(GeneratorOptions options) : options_(std::move(options)) {}

    GeneratorOptions options_;
    bool loaded_ = false;
    models::Qwen3Config config_;
    std::vector<Device> devices_;
    models::Qwen3DistributedWeights weights_;
    tokenizer::QwenTokenizer tokenizer_;
    std::vector<AttentionKVCache> attention_caches_;
    std::vector<DeltaNetCache> deltanet_caches_;

    ApiStatus load() {
        namespace fs = std::filesystem;
        if (options_.model_dir.empty()) {
            return {ApiStatusCode::InvalidArgument, "model_dir is empty"};
        }
        if (!fs::is_directory(options_.model_dir)) {
            return {ApiStatusCode::FileNotFound,
                    "model directory does not exist: " + options_.model_dir};
        }
        if (options_.max_devices <= 0) options_.max_devices = 1;
        InitializeBuiltinBackends();
        DeviceManager::instance().initialize();
        devices_ = select_gpu_devices(options_.backend, options_.max_devices);
        if (devices_.empty() && options_.backend == "cpu") {
            devices_.push_back(Device::Cpu());
        }
        if (devices_.empty()) {
            return {ApiStatusCode::InvalidDevice,
                    "no device is available for backend: " + options_.backend};
        }
        Status status = config_.load_json(
            (fs::path(options_.model_dir) / "config.json").string());
        if (!status.ok()) return {ApiStatusCode::InvalidModel, status.message()};
        models::Qwen3WeightLoader loader;
        status = loader.open(options_.model_dir, config_);
        if (!status.ok()) return {ApiStatusCode::InvalidModel, status.message()};
        status = loader.load_distributed(devices_, &weights_);
        if (!status.ok()) return {ApiStatusCode::Unknown, status.message()};
        status = tokenizer_.load(options_.model_dir);
        if (!status.ok()) return {ApiStatusCode::InvalidModel, status.message()};
        attention_caches_.resize(static_cast<size_t>(config_.num_hidden_layers));
        deltanet_caches_.resize(static_cast<size_t>(config_.num_hidden_layers));
        loaded_ = true;
        return {};
    }

    ApiStatus reset() {
        attention_caches_.assign(attention_caches_.size(), {});
        deltanet_caches_.assign(deltanet_caches_.size(), {});
        return {};
    }
};

Generator::Generator(GeneratorOptions options)
    : impl_(std::make_unique<Impl>(std::move(options))) {}
Generator::~Generator() = default;
Generator::Generator(Generator&&) noexcept = default;
Generator& Generator::operator=(Generator&&) noexcept = default;

ApiStatus Generator::load_model() { return impl_->load(); }
ApiStatus Generator::reset() { return impl_->reset(); }

ApiStatus Generator::generate(const GenerationOptions& options,
                              GenerationResult* result) {
    if (result == nullptr) {
        return {ApiStatusCode::InvalidArgument, "result is null"};
    }
    if (!impl_->loaded_) {
        return {ApiStatusCode::InvalidModel, "model has not been loaded"};
    }
    if (options.prompt.empty() || options.max_new_tokens <= 0) {
        return {ApiStatusCode::InvalidArgument,
                "prompt must be non-empty and max_new_tokens must be positive"};
    }
    ApiStatus status = impl_->reset();
    if (!status.ok()) return status;

    std::string prompt = options.prompt;
    if (options.use_chat_template) {
        prompt = impl_->tokenizer_.build_chat_prompt(
            {{"user", options.prompt}}, true, options.enable_thinking);
    }
    std::vector<int64_t> generated = impl_->tokenizer_.encode(prompt, true);
    const size_t prompt_count = generated.size();
    std::vector<int64_t> eos = impl_->config_.eos_token_ids;
    eos.insert(eos.end(), impl_->tokenizer_.eos_token_ids().begin(),
               impl_->tokenizer_.eos_token_ids().end());
    std::sort(eos.begin(), eos.end());
    eos.erase(std::unique(eos.begin(), eos.end()), eos.end());
    const int64_t max_cache_len = static_cast<int64_t>(prompt_count) +
                                  options.max_new_tokens;
    if (max_cache_len > impl_->config_.max_position_embeddings) {
        return {ApiStatusCode::InvalidArgument, "prompt exceeds model context"};
    }

    using Clock = std::chrono::steady_clock;
    const auto start = Clock::now();
    int64_t decode_count = 0;
    std::vector<double> decode_step_seconds;
    decode_step_seconds.reserve(static_cast<size_t>(options.max_new_tokens));
    double prefill_seconds = 0.0;
    double first_token_seconds = 0.0;
    Clock::time_point previous_token = start;
    const Device final_device = impl_->devices_.back();
    auto final_backend = BackendRegistry::instance().get_backend(final_device);
    if (final_backend == nullptr) {
        return {ApiStatusCode::BackendError,
                "failed to create final-device backend"};
    }
    std::shared_ptr<Buffer> argmax_buffer;
    if (final_device.is_gpu()) {
        argmax_buffer = final_backend->create_buffer(
            sizeof(int64_t), MemoryType::Device);
    }
    const bool profile_decode = [] {
        const char* value = std::getenv("HYBRIDAI_DECODE_PROFILE");
        return value != nullptr && value[0] == '1';
    }();
    std::vector<double> layer_profile_ms(
        profile_decode ? impl_->weights_.layers.size() : 0, 0.0);
    for (int32_t step = 0; step < options.max_new_tokens; ++step) {
        const std::vector<int64_t> ids =
            step == 0 ? generated : std::vector<int64_t>{generated.back()};
        Tensor embedded;
        auto first_backend = BackendRegistry::instance().get_backend(
            impl_->devices_.front());
        ::hybridai::Status internal_status = lookup_embedding(
            impl_->weights_.shared.embed_tokens, ids, first_backend.get(),
            impl_->devices_.front(), &embedded);
        if (!internal_status.ok()) {
            return {ApiStatusCode::BackendError, internal_status.message()};
        }
        Tensor hidden = std::move(embedded);
        for (size_t li = 0; li < impl_->weights_.layers.size(); ++li) {
            const auto layer_start = Clock::now();
            const auto& layer = impl_->weights_.layers[li];
            const Device device = impl_->weights_.layer_devices[li];
            auto backend = BackendRegistry::instance().get_backend(device);
            if (hidden.device() != device) hidden = hidden.to(device);
            Tensor normalized = qwen_rmsnorm_reference(
                hidden, layer.input_layernorm, backend.get(), device,
                impl_->config_.rms_norm_eps);
            if (normalized.buffer() == nullptr) {
                return {ApiStatusCode::BackendError,
                        "input RMSNorm failed at layer " + std::to_string(li)};
            }
            Tensor mixed = layer.is_attention_layer
                ? qwen_attention_reference(normalized, layer, backend.get(),
                    device, impl_->config_, &impl_->attention_caches_[li],
                    max_cache_len)
                : qwen_deltanet_reference(normalized, layer, backend.get(),
                    device, impl_->config_, &impl_->deltanet_caches_[li]);
            if (mixed.buffer() == nullptr) {
                return {ApiStatusCode::BackendError, "mixer forward failed"};
            }
            hidden = make_residual(hidden, mixed, backend.get(), device);
            Tensor norm = qwen_rmsnorm_reference(
                hidden, layer.post_attention_layernorm, backend.get(), device,
                impl_->config_.rms_norm_eps);
            Tensor mlp = qwen_mlp_reference(norm, layer, backend.get(), device);
            hidden = make_residual(hidden, mlp, backend.get(), device);
            if (hidden.buffer() == nullptr) {
                return {ApiStatusCode::BackendError, "layer forward failed"};
            }
            if (profile_decode) {
                internal_status = backend->synchronize();
                if (!internal_status.ok()) {
                    return {ApiStatusCode::BackendError,
                            "profile synchronization failed at layer " +
                                std::to_string(li)};
                }
                if (step > 0) {
                    layer_profile_ms[li] += std::chrono::duration<double,
                        std::milli>(Clock::now() - layer_start).count();
                }
            }
        }
        if (hidden.device() != final_device) hidden = hidden.to(final_device);
        Tensor final_hidden = qwen_rmsnorm_reference(
            hidden, impl_->weights_.shared.final_norm, final_backend.get(),
            final_device, impl_->config_.rms_norm_eps);
        Tensor logits = Linear::forward(final_hidden, impl_->weights_.shared.lm_head,
                                        true, nullptr);
        if (logits.buffer() == nullptr) {
            return {ApiStatusCode::BackendError, "lm_head forward failed"};
        }
        int64_t next = 0;
        bool used_device_argmax = false;
        if (logits.device().is_gpu() && argmax_buffer != nullptr) {
            internal_status = final_backend->argmax_last_row(
                argmax_buffer.get(), logits.buffer().get(), logits.dtype(),
                logits.shape().dim(0), logits.shape().dim(1));
            if (internal_status.ok()) {
                internal_status = final_backend->memcpy_d2h(
                    &next, argmax_buffer.get(), sizeof(next));
                if (internal_status.ok()) {
                    // The D2H copy is enqueued after the GEMM, bias and
                    // argmax kernels on the same default stream. Synchronize
                    // only after the copy, immediately before consuming the
                    // host value, instead of synchronizing once before
                    // argmax and again implicitly at the host boundary.
                    internal_status = final_backend->synchronize();
                    used_device_argmax = internal_status.ok();
                }
            }
        }
        if (!used_device_argmax) {
            const size_t count = static_cast<size_t>(logits.numel());
            const size_t offset = static_cast<size_t>(logits.shape().dim(0) - 1) *
                                  static_cast<size_t>(logits.shape().dim(1));
            float best = -std::numeric_limits<float>::infinity();
            if (logits.dtype() == DType::FP32) {
                std::vector<float> host(count);
                internal_status = final_backend->memcpy_d2h(
                    host.data(), logits.buffer().get(), logits.nbytes());
                if (!internal_status.ok()) {
                    return {ApiStatusCode::BackendError,
                            internal_status.message()};
                }
                internal_status = final_backend->synchronize();
                if (!internal_status.ok()) {
                    return {ApiStatusCode::BackendError,
                            internal_status.message()};
                }
                for (int64_t id = 0; id < logits.shape().dim(1); ++id) {
                    const float value = host[offset + static_cast<size_t>(id)];
                    if (value > best) { best = value; next = id; }
                }
            } else if (logits.dtype() == DType::BF16) {
                std::vector<uint16_t> host(count);
                internal_status = final_backend->memcpy_d2h(
                    host.data(), logits.buffer().get(), logits.nbytes());
                if (!internal_status.ok()) {
                    return {ApiStatusCode::BackendError,
                            internal_status.message()};
                }
                for (int64_t id = 0; id < logits.shape().dim(1); ++id) {
                    uint32_t bits = static_cast<uint32_t>(
                        host[offset + static_cast<size_t>(id)]) << 16;
                    float value = 0.0f;
                    std::memcpy(&value, &bits, sizeof(value));
                    if (value > best) { best = value; next = id; }
                }
            } else {
                return {ApiStatusCode::BackendError,
                        "unsupported logits dtype for argmax"};
            }
        }
        generated.push_back(next);
        const auto token_completed = Clock::now();
        if (step == 0) {
            prefill_seconds = std::chrono::duration<double>(
                token_completed - start).count();
            first_token_seconds = prefill_seconds;
        } else {
            ++decode_count;
            decode_step_seconds.push_back(std::chrono::duration<double>(
                token_completed - previous_token).count());
        }
        previous_token = token_completed;
        if (std::binary_search(eos.begin(), eos.end(), next)) break;
    }
    const auto suffix = std::vector<int64_t>(generated.begin() +
                                             static_cast<ptrdiff_t>(prompt_count),
                                             generated.end());
    result->token_ids = suffix;
    result->text = impl_->tokenizer_.decode(suffix, true);
    result->prompt_tokens = static_cast<int64_t>(prompt_count);
    result->elapsed_seconds = std::chrono::duration<double>(Clock::now() - start).count();
    result->prefill_seconds = prefill_seconds;
    result->time_to_first_token_seconds = first_token_seconds;
    result->decode_step_seconds = std::move(decode_step_seconds);
    result->decode_seconds = 0.0;
    for (double seconds : result->decode_step_seconds) {
        result->decode_seconds += seconds;
    }
    result->decode_tokens_per_second = result->decode_seconds > 0.0
        ? static_cast<double>(decode_count) / result->decode_seconds : 0.0;
    if (profile_decode && decode_count > 0) {
        std::cout << "[Decode profile] average layer wall time over "
                  << decode_count << " decode tokens (ms):" << std::endl;
        for (size_t li = 0; li < layer_profile_ms.size(); ++li) {
            const auto& layer = impl_->weights_.layers[li];
            std::cout << "  layer " << li << " "
                      << (layer.is_attention_layer ? "attention" : "deltanet")
                      << ": " << (layer_profile_ms[li] /
                                     static_cast<double>(decode_count))
                      << std::endl;
        }
    }
    return {};
}

} // namespace hybridai