#include "models/qwen_reference.h"

#include "backends/backend_registry.h"
#include "core/device_manager.h"
#include "ops/linear.h"
#include "tokenizer/qwen_tokenizer.h"

#include "hybrid.h"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>

namespace hybridai {

using namespace ops;

namespace {

Tensor concat_mtp_inputs(const Tensor& embedding, const Tensor& hidden,
                         Backend* backend, const Device& device) {
    if (backend == nullptr || embedding.buffer() == nullptr ||
        hidden.buffer() == nullptr || embedding.device() != device ||
        hidden.device() != device || embedding.dtype() != hidden.dtype() ||
        embedding.shape().ndim() != 2 || hidden.shape().ndim() != 2 ||
        embedding.shape().dim(0) != hidden.shape().dim(0)) {
        return Tensor();
    }
    const int64_t rows = embedding.shape().dim(0);
    const int64_t embedding_width = embedding.shape().dim(1);
    const int64_t hidden_width = hidden.shape().dim(1);
    auto buffer = backend->create_buffer(
        static_cast<size_t>(rows * (embedding_width + hidden_width)) *
            SizeOfDType(embedding.dtype()),
        embedding.buffer()->memory_type());
    if (buffer == nullptr) return Tensor();

    const size_t element_bytes = SizeOfDType(embedding.dtype());
    const size_t embedding_row_bytes =
        static_cast<size_t>(embedding_width) * element_bytes;
    const size_t hidden_row_bytes =
        static_cast<size_t>(hidden_width) * element_bytes;
    const size_t combined_row_bytes = embedding_row_bytes + hidden_row_bytes;
    for (int64_t row = 0; row < rows; ++row) {
        const size_t source_embedding_offset =
            static_cast<size_t>(row) * embedding_row_bytes;
        const size_t source_hidden_offset =
            static_cast<size_t>(row) * hidden_row_bytes;
        const size_t destination_offset =
            static_cast<size_t>(row) * combined_row_bytes;
        Status status = backend->copy_to_offset(
            buffer.get(), destination_offset, embedding.buffer().get(),
            source_embedding_offset, embedding_row_bytes);
        if (!status.ok()) return Tensor();
        status = backend->copy_to_offset(
            buffer.get(), destination_offset + embedding_row_bytes,
            hidden.buffer().get(), source_hidden_offset, hidden_row_bytes);
        if (!status.ok()) return Tensor();
    }
    return Tensor(Shape{rows, embedding_width + hidden_width},
                  embedding.dtype(), device, std::move(buffer));
}

Tensor row_copy(const Tensor& input, int64_t row, Backend* backend,
                const Device& device) {
    if (backend == nullptr || input.buffer() == nullptr ||
        input.shape().ndim() != 2 || input.shape().dim(0) <= 0 ||
        row < 0 || row >= input.shape().dim(0) || input.device() != device) {
        return Tensor();
    }
    const int64_t width = input.shape().dim(1);
    auto buffer = backend->create_buffer(input.nbytes() /
                                             static_cast<size_t>(input.shape().dim(0)),
                                         input.buffer()->memory_type());
    if (buffer == nullptr) return Tensor();
    const size_t row_bytes = static_cast<size_t>(width) * SizeOfDType(input.dtype());
    const size_t offset = static_cast<size_t>(row) * row_bytes;
    if (!backend->copy_to_offset(buffer.get(), 0, input.buffer().get(), offset,
                                 row_bytes).ok()) {
        return Tensor();
    }
    return Tensor(Shape{1, width}, input.dtype(), device, std::move(buffer));
}

Tensor last_row_copy(const Tensor& input, Backend* backend,
                     const Device& device) {
    if (input.shape().ndim() != 2 || input.shape().dim(0) <= 0) {
        return Tensor();
    }
    return row_copy(input, input.shape().dim(0) - 1, backend, device);
}

bool last_row_copy_into(const Tensor& input, Backend* backend,
                        const Device& device,
                        const std::shared_ptr<Buffer>& output_buffer,
                        Tensor* output) {
    if (output == nullptr || backend == nullptr || output_buffer == nullptr ||
        input.buffer() == nullptr || input.shape().ndim() != 2 ||
        input.shape().dim(0) <= 0 || input.device() != device) {
        return false;
    }
    const int64_t width = input.shape().dim(1);
    const size_t row_bytes = static_cast<size_t>(width) *
                             SizeOfDType(input.dtype());
    if (output_buffer->size() < row_bytes ||
        output_buffer->memory_type() != input.buffer()->memory_type()) {
        return false;
    }
    const size_t offset = static_cast<size_t>(input.shape().dim(0) - 1) *
                          row_bytes;
    if (!backend->copy_to_offset(output_buffer.get(), 0,
                                 input.buffer().get(), offset,
                                 row_bytes).ok()) {
        return false;
    }
    *output = Tensor(Shape{1, width}, input.dtype(), device, output_buffer);
    return true;
}

size_t speculative_proposal_width() {
    constexpr size_t kDefaultWidth = 8;
    constexpr size_t kMaxWidth = 16;
    const char* value = std::getenv("HYBRIDAI_SPECULATIVE_WIDTH");
    if (value == nullptr || value[0] == '\0') return kDefaultWidth;

    char* end = nullptr;
    const unsigned long parsed = std::strtoul(value, &end, 10);
    if (end == value || *end != '\0' || parsed == 0 ||
        parsed > kMaxWidth) {
        return kDefaultWidth;
    }
    return static_cast<size_t>(parsed);
}

Tensor qwen_mtp_forward_reference(
    const Tensor& embedding, const Tensor& hidden,
    const models::Qwen3MTPWeights& mtp, const models::Qwen3Config& config,
    Backend* backend, const Device& device,
    std::vector<AttentionKVCache>* attention_caches, int64_t max_cache_len) {
    if (backend == nullptr || attention_caches == nullptr ||
        mtp.layers.empty() || mtp.fc.buffer() == nullptr || mtp.norm.buffer() == nullptr) {
        return Tensor();
    }
    if (embedding.device() != device || hidden.device() != device ||
        embedding.shape() != hidden.shape()) {
        return Tensor();
    }
    Tensor normalized_embedding = qwen_rmsnorm_reference(
        embedding, mtp.pre_fc_norm_embedding, backend, device,
        config.rms_norm_eps);
    Tensor normalized_hidden = qwen_rmsnorm_reference(
        hidden, mtp.pre_fc_norm_hidden, backend, device, config.rms_norm_eps);
    Tensor combined = concat_mtp_inputs(normalized_embedding, normalized_hidden,
                                         backend, device);
    if (combined.buffer() == nullptr) return Tensor();
    Tensor current = Linear::forward(combined, mtp.fc, true, nullptr);
    if (current.buffer() == nullptr) return Tensor();

    attention_caches->resize(mtp.layers.size());
    for (size_t index = 0; index < mtp.layers.size(); ++index) {
        const auto& layer = mtp.layers[index];
        Tensor normalized = qwen_rmsnorm_reference(
            current, layer.input_layernorm, backend, device,
            config.rms_norm_eps);
        Tensor mixed = qwen_attention_reference(
            normalized, layer, backend, device, config,
            &(*attention_caches)[index], max_cache_len);
        if (mixed.buffer() == nullptr) return Tensor();
        current = make_residual(current, mixed, backend, device);
        Tensor post_norm = qwen_rmsnorm_reference(
            current, layer.post_attention_layernorm, backend, device,
            config.rms_norm_eps);
        Tensor mlp = qwen_mlp_reference(post_norm, layer, backend, device);
        current = make_residual(current, mlp, backend, device);
        if (current.buffer() == nullptr) return Tensor();
    }
    return qwen_rmsnorm_reference(current, mtp.norm, backend, device,
                                  config.rms_norm_eps);
}

bool clone_attention_caches(const std::vector<AttentionKVCache>& source,
                            std::vector<AttentionKVCache>* destination) {
    if (destination == nullptr) return false;
    destination->resize(source.size());
    for (size_t index = 0; index < source.size(); ++index) {
        if (!source[index].initialized()) {
            (*destination)[index] = AttentionKVCache{};
            continue;
        }
        if (!source[index].clone(&(*destination)[index]).ok()) return false;
    }
    return true;
}

void swap_attention_caches(std::vector<AttentionKVCache>* lhs,
                           std::vector<AttentionKVCache>* rhs) noexcept {
    if (lhs == nullptr || rhs == nullptr || lhs->size() != rhs->size()) return;
    for (size_t index = 0; index < lhs->size(); ++index) {
        (*lhs)[index].swap(&(*rhs)[index]);
    }
}

struct DecoderCacheSnapshot {
    std::vector<AttentionKVCache> attention;
    std::vector<DeltaNetCache> deltanet;
};

bool clone_decoder_caches(const std::vector<AttentionKVCache>& attention,
                          const std::vector<DeltaNetCache>& deltanet,
                          DecoderCacheSnapshot* snapshot) {
    if (snapshot == nullptr || attention.size() != deltanet.size()) return false;
    snapshot->attention.resize(attention.size());
    snapshot->deltanet.resize(deltanet.size());
    for (size_t index = 0; index < attention.size(); ++index) {
        if (!attention[index].initialized()) {
            snapshot->attention[index] = AttentionKVCache{};
        } else if (!attention[index].clone(&snapshot->attention[index]).ok()) {
            return false;
        }
        if (!deltanet[index].initialized()) {
            snapshot->deltanet[index] = DeltaNetCache{};
        } else if (!deltanet[index].checkpoint(
                       &snapshot->deltanet[index]).ok()) {
            return false;
        }
    }
    return true;
}

void swap_decoder_caches(std::vector<AttentionKVCache>* attention,
                         std::vector<DeltaNetCache>* deltanet,
                         DecoderCacheSnapshot* snapshot) noexcept {
    if (attention == nullptr || deltanet == nullptr || snapshot == nullptr ||
        attention->size() != snapshot->attention.size() ||
        deltanet->size() != snapshot->deltanet.size()) return;
    swap_attention_caches(attention, &snapshot->attention);
    for (size_t index = 0; index < deltanet->size(); ++index) {
        (*deltanet)[index].swap(&snapshot->deltanet[index]);
    }
}

bool argmax_last_row_reference(const Tensor& logits, Backend* backend,
                               std::shared_ptr<Buffer>* argmax_buffer,
                               int64_t* result) {
    if (backend == nullptr || result == nullptr || logits.buffer() == nullptr ||
        logits.shape().ndim() != 2) return false;
    if (logits.device().is_gpu() && argmax_buffer != nullptr) {
        if (*argmax_buffer == nullptr) {
            *argmax_buffer = backend->create_buffer(sizeof(int64_t),
                                                     MemoryType::Device);
        }
        if (*argmax_buffer != nullptr) {
            Status status = backend->argmax_last_row(
                argmax_buffer->get(), logits.buffer().get(), logits.dtype(),
                logits.shape().dim(0), logits.shape().dim(1));
            if (status.ok() && backend->memcpy_d2h(
                    result, argmax_buffer->get(), sizeof(*result)).ok() &&
                backend->synchronize().ok()) {
                return true;
            }
        }
    }
    const size_t offset = static_cast<size_t>(logits.shape().dim(0) - 1) *
                          static_cast<size_t>(logits.shape().dim(1));
    float best = -std::numeric_limits<float>::infinity();
    if (logits.dtype() == DType::FP32) {
        std::vector<float> host(static_cast<size_t>(logits.numel()));
        if (!backend->memcpy_d2h(host.data(), logits.buffer().get(),
                                  logits.nbytes()).ok() ||
            !backend->synchronize().ok()) return false;
        for (int64_t id = 0; id < logits.shape().dim(1); ++id) {
            const float value = host[offset + static_cast<size_t>(id)];
            if (value > best) { best = value; *result = id; }
        }
        return true;
    }
    if (logits.dtype() == DType::BF16) {
        std::vector<uint16_t> host(static_cast<size_t>(logits.numel()));
        if (!backend->memcpy_d2h(host.data(), logits.buffer().get(),
                                  logits.nbytes()).ok() ||
            !backend->synchronize().ok()) return false;
        for (int64_t id = 0; id < logits.shape().dim(1); ++id) {
            uint32_t bits = static_cast<uint32_t>(
                host[offset + static_cast<size_t>(id)]) << 16;
            float value = 0.0f;
            std::memcpy(&value, &bits, sizeof(value));
            if (value > best) { best = value; *result = id; }
        }
        return true;
    }
    return false;
}

bool argmax_rows_reference(const Tensor& logits, Backend* backend,
                           std::shared_ptr<Buffer>* argmax_buffer,
                           std::vector<int64_t>* results) {
    if (backend == nullptr || results == nullptr || logits.buffer() == nullptr ||
        logits.shape().ndim() != 2) return false;
    const int64_t rows = logits.shape().dim(0);
    const int64_t columns = logits.shape().dim(1);
    results->assign(static_cast<size_t>(rows), 0);
    if (logits.device().is_gpu()) {
        if (argmax_buffer == nullptr || *argmax_buffer == nullptr ||
            (*argmax_buffer)->size() <
                static_cast<size_t>(rows) * sizeof(int64_t)) {
            if (argmax_buffer == nullptr) return false;
            *argmax_buffer = backend->create_buffer(
                static_cast<size_t>(rows) * sizeof(int64_t),
                MemoryType::Device);
        }
        if (argmax_buffer != nullptr && *argmax_buffer != nullptr) {
            Status status = backend->argmax_rows(
                argmax_buffer->get(), logits.buffer().get(), logits.dtype(), rows,
                columns);
            if (status.ok() && backend->memcpy_d2h(
                    results->data(), argmax_buffer->get(),
                    static_cast<size_t>(rows) * sizeof(int64_t)).ok() &&
                backend->synchronize().ok()) {
                return true;
            }
        }
    }
    if (logits.dtype() == DType::FP32) {
        std::vector<float> host(static_cast<size_t>(logits.numel()));
        if (!backend->memcpy_d2h(host.data(), logits.buffer().get(),
                                 logits.nbytes()).ok() ||
            !backend->synchronize().ok()) return false;
        for (int64_t row = 0; row < rows; ++row) {
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t column = 0; column < columns; ++column) {
                const float value = host[static_cast<size_t>(row * columns + column)];
                if (value > best) {
                    best = value;
                    (*results)[static_cast<size_t>(row)] = column;
                }
            }
        }
        return true;
    }
    if (logits.dtype() == DType::BF16) {
        std::vector<uint16_t> host(static_cast<size_t>(logits.numel()));
        if (!backend->memcpy_d2h(host.data(), logits.buffer().get(),
                                 logits.nbytes()).ok() ||
            !backend->synchronize().ok()) return false;
        for (int64_t row = 0; row < rows; ++row) {
            float best = -std::numeric_limits<float>::infinity();
            for (int64_t column = 0; column < columns; ++column) {
                uint32_t bits = static_cast<uint32_t>(
                    host[static_cast<size_t>(row * columns + column)]) << 16;
                float value = 0.0f;
                std::memcpy(&value, &bits, sizeof(value));
                if (value > best) {
                    best = value;
                    (*results)[static_cast<size_t>(row)] = column;
                }
            }
        }
        return true;
    }
    return false;
}

} // namespace

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
    std::vector<AttentionKVCache> mtp_attention_caches_;
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
        if (options_.enable_mtp) {
            status = loader.load_mtp(devices_.back(), &weights_.mtp);
            if (!status.ok()) return {ApiStatusCode::InvalidModel,
                                      status.message()};
        }
        status = tokenizer_.load(options_.model_dir);
        if (!status.ok()) return {ApiStatusCode::InvalidModel, status.message()};
        attention_caches_.resize(static_cast<size_t>(config_.num_hidden_layers));
        mtp_attention_caches_.resize(
            static_cast<size_t>(config_.mtp_num_hidden_layers));
        deltanet_caches_.resize(static_cast<size_t>(config_.num_hidden_layers));
        loaded_ = true;
        return {};
    }

    ApiStatus reset() {
        attention_caches_.assign(attention_caches_.size(), {});
        mtp_attention_caches_.assign(mtp_attention_caches_.size(), {});
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
    if (options.enable_speculative_mtp && !impl_->options_.enable_mtp) {
        return {ApiStatusCode::InvalidArgument,
                "speculative MTP requires GeneratorOptions::enable_mtp"};
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
    int64_t mtp_proposed_tokens = 0;
    int64_t mtp_accepted_tokens = 0;
    int64_t mtp_rejected_tokens = 0;
    int64_t mtp_correction_tokens = 0;
    int64_t speculative_replay_tokens = 0;
    int64_t speculative_mtp_recovery_tokens = 0;
    int64_t speculative_max_proposal_width = 0;
    int64_t speculative_mtp_cache_clone_count = 0;
    int64_t speculative_target_cache_clone_count = 0;
    int64_t mtp_fallback_steps = 0;
    double speculative_proposal_seconds = 0.0;
    double speculative_verification_seconds = 0.0;
    double speculative_replay_seconds = 0.0;
    double speculative_fallback_seconds = 0.0;
    double speculative_mtp_cache_clone_seconds = 0.0;
    double speculative_target_cache_clone_seconds = 0.0;
    double speculative_argmax_seconds = 0.0;
    int64_t speculative_rounds = 0;
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
    std::shared_ptr<Buffer> argmax_rows_buffer;
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
    const bool profile_speculative_verification = [] {
        const char* value = std::getenv("HYBRIDAI_SPECULATIVE_PROFILE");
        return value != nullptr && value[0] == '1';
    }();
    std::vector<double> speculative_verification_layer_ms(
        profile_speculative_verification ? impl_->weights_.layers.size() : 0,
        0.0);
    int64_t speculative_verification_profile_tokens = 0;
    double speculative_verification_final_norm_ms = 0.0;
    double speculative_verification_lm_head_ms = 0.0;
    double speculative_verification_attention_ms = 0.0;
    double speculative_verification_deltanet_ms = 0.0;
    double speculative_verification_transfer_ms = 0.0;

    // This path uses fixed-width greedy speculative decoding. The target
    // cache is committed only after verification; rejected suffixes are
    // discarded and the accepted prefix plus target correction is replayed.
    if (options.enable_speculative_mtp &&
        !impl_->weights_.mtp.layers.empty()) {
        auto target_forward = [&](const std::vector<int64_t>& ids,
                                  std::vector<AttentionKVCache>* attention,
                                  std::vector<DeltaNetCache>* deltanet,
                                  bool profile_layers = false)
            -> std::pair<Tensor, Tensor> {
            auto first_backend = BackendRegistry::instance().get_backend(
                impl_->devices_.front());
            Tensor embedded;
            auto status = lookup_embedding(
                impl_->weights_.shared.embed_tokens, ids, first_backend.get(),
                impl_->devices_.front(), &embedded);
            if (!status.ok()) return {};
            Tensor hidden = embedded;
            for (size_t li = 0; li < impl_->weights_.layers.size(); ++li) {
                const auto layer_start = Clock::now();
                const auto& layer = impl_->weights_.layers[li];
                const Device device = impl_->weights_.layer_devices[li];
                auto backend = BackendRegistry::instance().get_backend(device);
                if (hidden.device() != device) {
                    const auto transfer_start = Clock::now();
                    hidden = hidden.to(device);
                    if (profile_layers) {
                        speculative_verification_transfer_ms +=
                            std::chrono::duration<double, std::milli>(
                                Clock::now() - transfer_start).count();
                    }
                }
                Tensor normalized = qwen_rmsnorm_reference(
                    hidden, layer.input_layernorm, backend.get(), device,
                    impl_->config_.rms_norm_eps);
                Tensor mixed = layer.is_attention_layer
                    ? qwen_attention_reference(
                          normalized, layer, backend.get(), device,
                          impl_->config_, &(*attention)[li], max_cache_len)
                    : qwen_deltanet_reference(
                          normalized, layer, backend.get(), device,
                          impl_->config_, &(*deltanet)[li]);
                if (mixed.buffer() == nullptr) return {};
                hidden = make_residual(hidden, mixed, backend.get(), device);
                Tensor norm = qwen_rmsnorm_reference(
                    hidden, layer.post_attention_layernorm, backend.get(),
                    device, impl_->config_.rms_norm_eps);
                Tensor mlp = qwen_mlp_reference(
                    norm, layer, backend.get(), device);
                hidden = make_residual(hidden, mlp, backend.get(), device);
                if (hidden.buffer() == nullptr) return {};
                if (profile_layers) {
                    if (!backend->synchronize().ok()) return {};
                    const double layer_ms =
                        std::chrono::duration<double, std::milli>(
                            Clock::now() - layer_start).count();
                    speculative_verification_layer_ms[li] += layer_ms;
                    if (layer.is_attention_layer) {
                        speculative_verification_attention_ms += layer_ms;
                    } else {
                        speculative_verification_deltanet_ms += layer_ms;
                    }
                }
            }
            if (profile_layers) ++speculative_verification_profile_tokens;
            if (hidden.device() != final_device) hidden = hidden.to(final_device);
            const auto final_norm_start = Clock::now();
            Tensor final_hidden = qwen_rmsnorm_reference(
                hidden, impl_->weights_.shared.final_norm, final_backend.get(),
                final_device, impl_->config_.rms_norm_eps);
            if (profile_layers) {
                if (!final_backend->synchronize().ok()) return {};
                speculative_verification_final_norm_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - final_norm_start).count();
            }
            const auto lm_head_start = Clock::now();
            Tensor logits = Linear::forward(
                final_hidden, impl_->weights_.shared.lm_head, true, nullptr);
            if (profile_layers) {
                if (!final_backend->synchronize().ok()) return {};
                speculative_verification_lm_head_ms +=
                    std::chrono::duration<double, std::milli>(
                        Clock::now() - lm_head_start).count();
            }
            if (final_hidden.buffer() == nullptr || logits.buffer() == nullptr) {
                return {};
            }
            return {std::move(final_hidden), std::move(logits)};
        };

        auto append_token = [&](int64_t token, const Clock::time_point& done) {
            generated.push_back(token);
            ++decode_count;
            decode_step_seconds.push_back(std::chrono::duration<double>(
                done - previous_token).count());
            previous_token = done;
        };

        // Multi-token greedy speculative decoding.  y1 is produced by the
        // committed target cache; MTP recursively proposes the following
        // tokens, and the target verifies the whole proposal in one forward.
        {
            auto initial = target_forward(
                generated, &impl_->attention_caches_, &impl_->deltanet_caches_);
            if (initial.second.buffer() == nullptr) {
                return {ApiStatusCode::BackendError,
                        "speculative prefill forward failed"};
            }
            int64_t next_target = 0;
            if (!argmax_last_row_reference(initial.second, final_backend.get(),
                                           &argmax_buffer, &next_target)) {
                return {ApiStatusCode::BackendError,
                        "failed to compute speculative target argmax"};
            }
            prefill_seconds = std::chrono::duration<double>(
                Clock::now() - start).count();
            first_token_seconds = prefill_seconds;
            Tensor target_hidden = last_row_copy(
                initial.first, final_backend.get(), final_device);
            if (target_hidden.buffer() == nullptr) {
                return {ApiStatusCode::BackendError,
                        "failed to extract speculative hidden row"};
            }

            // MTP attention uses cache.length as its RoPE position offset.
            // Seed it with the prompt rows so its first generated input uses
            // the same absolute position as the target model.
            {
                auto embedding_backend = BackendRegistry::instance().get_backend(
                    impl_->devices_.front());
                Tensor prompt_embedding;
                auto embedding_status = lookup_embedding(
                    impl_->weights_.shared.embed_tokens, generated,
                    embedding_backend.get(), impl_->devices_.front(),
                    &prompt_embedding);
                if (!embedding_status.ok()) {
                    return {ApiStatusCode::BackendError,
                            embedding_status.message()};
                }
                if (prompt_embedding.device() != final_device) {
                    prompt_embedding = prompt_embedding.to(final_device);
                }
                std::vector<AttentionKVCache> mtp_prompt_cache;
                Tensor mtp_prompt_hidden = qwen_mtp_forward_reference(
                    prompt_embedding, initial.first, impl_->weights_.mtp,
                    impl_->config_, final_backend.get(), final_device,
                    &mtp_prompt_cache, max_cache_len);
                if (mtp_prompt_hidden.buffer() == nullptr) {
                    return {ApiStatusCode::BackendError,
                            "speculative MTP prompt prefill failed"};
                }
                swap_attention_caches(&impl_->mtp_attention_caches_,
                                      &mtp_prompt_cache);
            }

            const size_t proposal_width = speculative_proposal_width();
            std::vector<AttentionKVCache> mtp_scratch;
            std::vector<std::vector<AttentionKVCache>> mtp_prefix_checkpoints;
            DecoderCacheSnapshot verification;
            auto embedding_backend = BackendRegistry::instance().get_backend(
                impl_->devices_.front());
            const int64_t hidden_size = impl_->config_.hidden_size;
            std::shared_ptr<Buffer> proposal_embedding_buffer;
            std::shared_ptr<Buffer> proposal_embedding_ids_buffer;
            std::shared_ptr<Buffer> proposal_hidden_buffer;
            if (embedding_backend != nullptr) {
                proposal_embedding_buffer = embedding_backend->create_buffer(
                    static_cast<size_t>(hidden_size) *
                        SizeOfDType(DType::BF16),
                    MemoryType::Device);
                proposal_embedding_ids_buffer = embedding_backend->create_buffer(
                    sizeof(int64_t), MemoryType::Device);
            }
            std::shared_ptr<Buffer> proposal_logits_buffer;
            while (static_cast<int32_t>(generated.size() - prompt_count) <
                   options.max_new_tokens) {
                const int32_t remaining = options.max_new_tokens -
                    static_cast<int32_t>(generated.size() - prompt_count);
                if (std::binary_search(eos.begin(), eos.end(), next_target) ||
                    remaining == 1) {
                    append_token(next_target, Clock::now());
                    break;
                }

                const size_t proposal_count = std::min<size_t>(
                    proposal_width, static_cast<size_t>(remaining - 1));
                speculative_max_proposal_width = std::max<int64_t>(
                    speculative_max_proposal_width,
                    static_cast<int64_t>(proposal_count));
                ++speculative_rounds;
                const auto proposal_start = Clock::now();
                const auto mtp_clone_start = Clock::now();
                if (!clone_attention_caches(impl_->mtp_attention_caches_,
                                            &mtp_scratch)) {
                    return {ApiStatusCode::BackendError,
                            "failed to clone MTP proposal cache"};
                }
                ++speculative_mtp_cache_clone_count;
                speculative_mtp_cache_clone_seconds +=
                    std::chrono::duration<double>(Clock::now() - mtp_clone_start)
                        .count();
                mtp_prefix_checkpoints.clear();
                mtp_prefix_checkpoints.emplace_back();
                const auto checkpoint_start = Clock::now();
                if (!clone_attention_caches(impl_->mtp_attention_caches_,
                                            &mtp_prefix_checkpoints.back())) {
                    return {ApiStatusCode::BackendError,
                            "failed to checkpoint MTP proposal prefix"};
                }
                ++speculative_mtp_cache_clone_count;
                speculative_mtp_cache_clone_seconds +=
                    std::chrono::duration<double>(Clock::now() - checkpoint_start)
                        .count();
                std::vector<int64_t> candidates;
                candidates.reserve(proposal_count);
                Tensor proposal_hidden = target_hidden;
                int64_t proposal_input = next_target;
                for (size_t index = 0; index < proposal_count;
                     ++index) {
                    Tensor candidate_embedding;
                    auto embedding_status = lookup_embedding_into(
                        impl_->weights_.shared.embed_tokens, {proposal_input},
                        embedding_backend.get(), impl_->devices_.front(),
                        proposal_embedding_buffer,
                        proposal_embedding_ids_buffer,
                        &candidate_embedding);
                    if (!embedding_status.ok()) {
                        return {ApiStatusCode::BackendError,
                                embedding_status.message()};
                    }
                    if (candidate_embedding.device() != final_device) {
                        candidate_embedding = candidate_embedding.to(final_device);
                    }
                    Tensor mtp_hidden = qwen_mtp_forward_reference(
                        candidate_embedding, proposal_hidden, impl_->weights_.mtp,
                        impl_->config_, final_backend.get(), final_device,
                        &mtp_scratch, max_cache_len);
                    if (mtp_hidden.buffer() == nullptr) {
                        return {ApiStatusCode::BackendError,
                                "speculative MTP proposal forward failed"};
                    }
                    const size_t logits_bytes =
                        static_cast<size_t>(mtp_hidden.shape().dim(0)) *
                        static_cast<size_t>(impl_->config_.vocab_size) *
                        SizeOfDType(mtp_hidden.dtype());
                    if (proposal_logits_buffer == nullptr ||
                        proposal_logits_buffer->device() != final_device ||
                        proposal_logits_buffer->memory_type() !=
                            mtp_hidden.buffer()->memory_type() ||
                        proposal_logits_buffer->size() < logits_bytes) {
                        proposal_logits_buffer = final_backend->create_buffer(
                            logits_bytes, mtp_hidden.buffer()->memory_type());
                    }
                    Tensor mtp_logits;
                    if (proposal_logits_buffer != nullptr) {
                        mtp_logits = Linear::forward_into(
                            mtp_hidden, impl_->weights_.shared.lm_head,
                            proposal_logits_buffer, true, nullptr);
                    }
                    int64_t candidate = 0;
                    if (mtp_logits.buffer() == nullptr ||
                        !argmax_last_row_reference(mtp_logits, final_backend.get(),
                                                   &argmax_buffer, &candidate)) {
                        return {ApiStatusCode::BackendError,
                                "failed to compute speculative candidate"};
                    }
                    candidates.push_back(candidate);
                    proposal_input = candidate;
                    if (proposal_hidden_buffer == nullptr ||
                        proposal_hidden_buffer->size() <
                            mtp_hidden.nbytes() /
                                static_cast<size_t>(mtp_hidden.shape().dim(0)) ||
                        proposal_hidden_buffer->memory_type() !=
                            mtp_hidden.buffer()->memory_type()) {
                        proposal_hidden_buffer = final_backend->create_buffer(
                            mtp_hidden.nbytes() /
                                static_cast<size_t>(mtp_hidden.shape().dim(0)),
                            mtp_hidden.buffer()->memory_type());
                    }
                    Tensor next_proposal_hidden;
                    if (!last_row_copy_into(
                            mtp_hidden, final_backend.get(), final_device,
                            proposal_hidden_buffer, &next_proposal_hidden)) {
                        return {ApiStatusCode::BackendError,
                                "failed to reuse speculative hidden workspace"};
                    }
                    proposal_hidden = std::move(next_proposal_hidden);
                    ++mtp_proposed_tokens;
                    mtp_prefix_checkpoints.emplace_back();
                    const auto prefix_checkpoint_start = Clock::now();
                    if (!clone_attention_caches(
                            mtp_scratch, &mtp_prefix_checkpoints.back())) {
                        return {ApiStatusCode::BackendError,
                                "failed to checkpoint MTP accepted prefix"};
                    }
                    ++speculative_mtp_cache_clone_count;
                    speculative_mtp_cache_clone_seconds +=
                        std::chrono::duration<double>(
                            Clock::now() - prefix_checkpoint_start)
                            .count();
                    if (std::binary_search(eos.begin(), eos.end(), candidate)) {
                        break;
                    }
                }
                speculative_proposal_seconds += std::chrono::duration<double>(
                    Clock::now() - proposal_start).count();

                const auto verification_start = Clock::now();
                const auto target_clone_start = Clock::now();
                // The verification cache is a complete checkpoint of both
                // target cache families. DeltaNet must be checkpointed as
                // state tensors, not merely rewound by length.
                if (!clone_decoder_caches(impl_->attention_caches_,
                                           impl_->deltanet_caches_,
                                           &verification)) {
                    return {ApiStatusCode::BackendError,
                            "failed to clone target verification cache"};
                }
                ++speculative_target_cache_clone_count;
                speculative_target_cache_clone_seconds +=
                    std::chrono::duration<double>(Clock::now() - target_clone_start)
                        .count();
                std::vector<int64_t> verification_ids{next_target};
                verification_ids.insert(verification_ids.end(),
                                        candidates.begin(), candidates.end());
                // Attention cached_gqa and DeltaNet grouped-conv/recurrent
                // kernels process rows in token order. Their multi-token path
                // is therefore causal-equivalent to repeated single-token
                // calls while avoiding one full layer traversal per token.
                auto verified = target_forward(
                    verification_ids, &verification.attention,
                    &verification.deltanet, profile_speculative_verification);
                if (verified.second.buffer() == nullptr) {
                    return {ApiStatusCode::BackendError,
                            "speculative target verification failed"};
                }
                // HIP launches are asynchronous.  Without an explicit
                // boundary here, the first argmax synchronization includes
                // the unfinished target verification and makes
                // speculative_argmax_seconds appear much larger than the
                // argmax itself.
                if (!final_backend->synchronize().ok()) {
                    return {ApiStatusCode::BackendError,
                            "speculative target verification synchronize failed"};
                }
                const auto argmax_start = Clock::now();
                std::vector<int64_t> target_predictions;
                if (!argmax_rows_reference(verified.second,
                                           final_backend.get(), &argmax_rows_buffer,
                                           &target_predictions)) {
                    return {ApiStatusCode::BackendError,
                            "speculative target verification argmax failed"};
                }
                speculative_argmax_seconds +=
                    std::chrono::duration<double>(Clock::now() - argmax_start)
                        .count();
                if (target_predictions.size() != verification_ids.size()) {
                    return {ApiStatusCode::BackendError,
                            "speculative target verification argmax row count mismatch"};
                }
                speculative_verification_seconds += std::chrono::duration<double>(
                    Clock::now() - verification_start).count();

                // verification_ids is [next_target, candidates...]. Row i of
                // target_predictions predicts candidates[i], while the last
                // row predicts the correction after the proposal.
                size_t accepted = 0;
                while (accepted < candidates.size() &&
                       target_predictions[accepted] == candidates[accepted]) {
                    ++accepted;
                }
                const bool fully_accepted = accepted == candidates.size();
                if (fully_accepted) {
                    swap_decoder_caches(&impl_->attention_caches_,
                                        &impl_->deltanet_caches_, &verification);
                    swap_attention_caches(&impl_->mtp_attention_caches_,
                                          &mtp_scratch);
                    append_token(next_target, Clock::now());
                    for (int64_t candidate : candidates) {
                        append_token(candidate, Clock::now());
                    }
                    mtp_accepted_tokens += static_cast<int64_t>(accepted);
                    next_target = target_predictions.back();
                    target_hidden = last_row_copy(
                        verified.first, final_backend.get(), final_device);
                    // The scratch cache was cloned from the persistent MTP
                    // history and extended in proposal order. On full
                    // acceptance it is the valid next MTP history; keep its
                    // cache position instead of resetting RoPE/cache state.
                } else {
                    const int64_t correction = target_predictions[accepted];
                    mtp_rejected_tokens += static_cast<int64_t>(
                        candidates.size() - accepted);
                    ++mtp_correction_tokens;
                    const auto replay_start = Clock::now();
                    std::vector<int64_t> replay_ids{next_target};
                    replay_ids.insert(replay_ids.end(), candidates.begin(),
                                      candidates.begin() +
                                          static_cast<ptrdiff_t>(accepted));
                    replay_ids.push_back(correction);
                    auto replay = target_forward(
                        replay_ids, &impl_->attention_caches_,
                        &impl_->deltanet_caches_);
                    speculative_replay_tokens +=
                        static_cast<int64_t>(replay_ids.size());
                    if (replay.second.buffer() == nullptr) {
                        return {ApiStatusCode::BackendError,
                                "speculative correction replay failed"};
                    }
                    append_token(next_target, Clock::now());
                    for (size_t index = 0; index < accepted; ++index)
                        append_token(candidates[index], Clock::now());
                    if (static_cast<int32_t>(generated.size() - prompt_count) <
                        options.max_new_tokens) {
                        append_token(correction, Clock::now());
                    }
                    mtp_accepted_tokens += static_cast<int64_t>(accepted);
                    if (static_cast<int32_t>(generated.size() - prompt_count) >=
                        options.max_new_tokens ||
                        std::binary_search(eos.begin(), eos.end(), correction)) {
                        speculative_replay_seconds += std::chrono::duration<double>(
                            Clock::now() - replay_start).count();
                        break;
                    }

                    speculative_replay_seconds += std::chrono::duration<double>(
                        Clock::now() - replay_start).count();
                    if (accepted >= mtp_prefix_checkpoints.size()) {
                        return {ApiStatusCode::BackendError,
                                "missing MTP accepted-prefix checkpoint"};
                    }
                    swap_attention_caches(
                        &impl_->mtp_attention_caches_,
                        &mtp_prefix_checkpoints[accepted]);
                    const size_t correction_input_index = replay_ids.size() - 2;
                    Tensor correction_input_hidden = row_copy(
                        replay.first, static_cast<int64_t>(correction_input_index),
                        final_backend.get(), final_device);
                    if (correction_input_hidden.buffer() == nullptr) {
                        return {ApiStatusCode::BackendError,
                                "failed to extract correction input hidden row"};
                    }
                    Tensor correction_embedding;
                    auto correction_embedding_status = lookup_embedding(
                        impl_->weights_.shared.embed_tokens, {correction},
                        embedding_backend.get(), impl_->devices_.front(),
                        &correction_embedding);
                    if (!correction_embedding_status.ok()) {
                        return {ApiStatusCode::BackendError,
                                correction_embedding_status.message()};
                    }
                    if (correction_embedding.device() != final_device) {
                        correction_embedding = correction_embedding.to(final_device);
                    }
                    Tensor recovered_hidden = qwen_mtp_forward_reference(
                        correction_embedding, correction_input_hidden,
                        impl_->weights_.mtp, impl_->config_, final_backend.get(),
                        final_device, &impl_->mtp_attention_caches_, max_cache_len);
                    if (recovered_hidden.buffer() == nullptr) {
                        return {ApiStatusCode::BackendError,
                                "MTP accepted-prefix recovery forward failed"};
                    }
                    ++speculative_mtp_recovery_tokens;
                    if (!argmax_last_row_reference(
                            replay.second, final_backend.get(), &argmax_buffer,
                            &next_target)) {
                        return {ApiStatusCode::BackendError,
                                "failed to compute speculative correction"};
                    }
                    target_hidden = last_row_copy(
                        replay.first, final_backend.get(), final_device);
                    if (target_hidden.buffer() == nullptr) {
                        return {ApiStatusCode::BackendError,
                                "failed to extract correction hidden row"};
                    }
                }
                if (static_cast<int32_t>(generated.size() - prompt_count) >=
                        options.max_new_tokens ||
                    (!generated.empty() && std::binary_search(
                        eos.begin(), eos.end(), generated.back()))) {
                    break;
                }
            }
            result->token_ids = std::vector<int64_t>(
                generated.begin() + static_cast<ptrdiff_t>(prompt_count),
                generated.end());
            result->text = impl_->tokenizer_.decode(result->token_ids, true);
            result->prompt_tokens = static_cast<int64_t>(prompt_count);
            result->elapsed_seconds = std::chrono::duration<double>(
                Clock::now() - start).count();
            result->prefill_seconds = prefill_seconds;
            result->time_to_first_token_seconds = first_token_seconds;
            result->decode_step_seconds = std::move(decode_step_seconds);
            result->decode_seconds = 0.0;
            for (double seconds : result->decode_step_seconds)
                result->decode_seconds += seconds;
            result->decode_tokens_per_second = result->decode_seconds > 0.0
                ? static_cast<double>(decode_count) / result->decode_seconds : 0.0;
            result->mtp_proposed_tokens = mtp_proposed_tokens;
            result->mtp_accepted_tokens = mtp_accepted_tokens;
            result->mtp_rejected_tokens = mtp_rejected_tokens;
            result->mtp_correction_tokens = mtp_correction_tokens;
            result->speculative_replay_tokens = speculative_replay_tokens;
            result->speculative_mtp_recovery_tokens =
                speculative_mtp_recovery_tokens;
            result->speculative_max_proposal_width =
                speculative_max_proposal_width;
            result->speculative_mtp_cache_clone_count =
                speculative_mtp_cache_clone_count;
            result->speculative_target_cache_clone_count =
                speculative_target_cache_clone_count;
            result->mtp_fallback_steps = mtp_fallback_steps;
            result->mtp_acceptance_rate = mtp_proposed_tokens > 0
                ? static_cast<double>(mtp_accepted_tokens) /
                      static_cast<double>(mtp_proposed_tokens) : 0.0;
            result->speculative_proposal_seconds = speculative_proposal_seconds;
            result->speculative_verification_seconds =
                speculative_verification_seconds;
            result->speculative_replay_seconds = speculative_replay_seconds;
            result->speculative_fallback_seconds = speculative_fallback_seconds;
            result->speculative_mtp_cache_clone_seconds =
                speculative_mtp_cache_clone_seconds;
            result->speculative_target_cache_clone_seconds =
                speculative_target_cache_clone_seconds;
            result->speculative_argmax_seconds = speculative_argmax_seconds;
            result->speculative_rounds = speculative_rounds;
            if (profile_speculative_verification &&
                speculative_verification_profile_tokens > 0) {
                std::cout << "[Speculative verification profile] average layer "
                          << "wall time over "
                          << speculative_verification_profile_tokens
                          << " verification forwards (ms):" << std::endl;
                for (size_t li = 0;
                     li < speculative_verification_layer_ms.size(); ++li) {
                    const auto& layer = impl_->weights_.layers[li];
                    std::cout << "  layer " << li << " "
                              << (layer.is_attention_layer ? "attention"
                                                            : "deltanet")
                              << ": "
                              << (speculative_verification_layer_ms[li] /
                                  static_cast<double>(
                                      speculative_verification_profile_tokens))
                              << std::endl;
                }
                std::cout << "  final_norm: "
                          << (speculative_verification_final_norm_ms /
                              static_cast<double>(
                                  speculative_verification_profile_tokens))
                          << std::endl;
                std::cout << "  lm_head: "
                          << (speculative_verification_lm_head_ms /
                              static_cast<double>(
                                  speculative_verification_profile_tokens))
                          << std::endl;
                std::cout << "  attention_total: "
                          << (speculative_verification_attention_ms /
                              static_cast<double>(
                                  speculative_verification_profile_tokens))
                          << std::endl;
                std::cout << "  deltanet_total: "
                          << (speculative_verification_deltanet_ms /
                              static_cast<double>(
                                  speculative_verification_profile_tokens))
                          << std::endl;
                std::cout << "  activation_transfer: "
                          << (speculative_verification_transfer_ms /
                              static_cast<double>(
                                  speculative_verification_profile_tokens))
                          << std::endl;
            }
            return {};
        }
    }
    for (int32_t step = 0; step < options.max_new_tokens; ++step) {
        DecoderCacheSnapshot target_scratch_caches;
        std::vector<AttentionKVCache>* active_attention_caches =
            &impl_->attention_caches_;
        std::vector<DeltaNetCache>* active_deltanet_caches =
            &impl_->deltanet_caches_;
        if (options.enable_speculative_mtp) {
            if (!clone_decoder_caches(impl_->attention_caches_,
                                      impl_->deltanet_caches_,
                                      &target_scratch_caches)) {
                return {ApiStatusCode::BackendError,
                        "failed to clone target decoder caches"};
            }
            active_attention_caches = &target_scratch_caches.attention;
            active_deltanet_caches = &target_scratch_caches.deltanet;
        }
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
        // Keep the embedding tensor available for the optional MTP branch.
        // Tensor copies share the buffer; this does not duplicate activation
        // storage and avoids moving the only reference out of `embedded`.
        Tensor hidden = embedded;
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
                    device, impl_->config_, &(*active_attention_caches)[li],
                    max_cache_len)
                : qwen_deltanet_reference(normalized, layer, backend.get(),
                    device, impl_->config_, &(*active_deltanet_caches)[li]);
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
        if (options.enable_speculative_mtp) {
            swap_decoder_caches(&impl_->attention_caches_,
                                &impl_->deltanet_caches_,
                                &target_scratch_caches);
        }
        Tensor mtp_logits;
        std::vector<AttentionKVCache> mtp_scratch_caches;
        if (impl_->options_.enable_mtp && !impl_->weights_.mtp.layers.empty()) {
            if (!clone_attention_caches(impl_->mtp_attention_caches_,
                                        &mtp_scratch_caches)) {
                return {ApiStatusCode::BackendError,
                        "failed to clone MTP attention cache"};
            }
            Tensor mtp_embedding = embedded;
            if (mtp_embedding.device() != final_device) {
                mtp_embedding = mtp_embedding.to(final_device);
            }
            Tensor mtp_hidden = qwen_mtp_forward_reference(
                mtp_embedding, hidden, impl_->weights_.mtp, impl_->config_,
                final_backend.get(), final_device,
                &mtp_scratch_caches, max_cache_len);
            if (mtp_hidden.buffer() == nullptr) {
                return {ApiStatusCode::BackendError,
                        "MTP forward failed before candidate verification"};
            }
            mtp_logits = Linear::forward(
                mtp_hidden, impl_->weights_.shared.lm_head, true, nullptr);
            if (mtp_logits.buffer() == nullptr) {
                return {ApiStatusCode::BackendError,
                        "MTP lm_head forward failed"};
            }
        }
        int64_t next = 0;
        if (!argmax_last_row_reference(logits, final_backend.get(),
                                       &argmax_buffer, &next)) {
            return {ApiStatusCode::BackendError,
                    "failed to compute target logits argmax"};
        }
        if (mtp_logits.buffer() != nullptr) {
            ++mtp_proposed_tokens;
            int64_t mtp_next = 0;
            if (!argmax_last_row_reference(mtp_logits, final_backend.get(),
                                           &argmax_buffer, &mtp_next)) {
                return {ApiStatusCode::BackendError,
                        "failed to compute MTP logits argmax"};
            }
                swap_attention_caches(&impl_->mtp_attention_caches_,
                      &mtp_scratch_caches);
            if (mtp_next == next) {
                ++mtp_accepted_tokens;
            } else {
                ++mtp_fallback_steps;
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
    result->mtp_proposed_tokens = mtp_proposed_tokens;
    result->mtp_accepted_tokens = mtp_accepted_tokens;
    result->mtp_rejected_tokens = mtp_rejected_tokens;
    result->mtp_correction_tokens = mtp_correction_tokens;
    result->speculative_replay_tokens = speculative_replay_tokens;
    result->speculative_mtp_recovery_tokens =
        speculative_mtp_recovery_tokens;
    result->speculative_max_proposal_width =
        speculative_max_proposal_width;
    result->speculative_mtp_cache_clone_count =
        speculative_mtp_cache_clone_count;
    result->speculative_target_cache_clone_count =
        speculative_target_cache_clone_count;
    result->mtp_fallback_steps = mtp_fallback_steps;
    result->mtp_acceptance_rate = mtp_proposed_tokens > 0
        ? static_cast<double>(mtp_accepted_tokens) /
              static_cast<double>(mtp_proposed_tokens)
        : 0.0;
    result->speculative_proposal_seconds = speculative_proposal_seconds;
    result->speculative_verification_seconds = speculative_verification_seconds;
    result->speculative_replay_seconds = speculative_replay_seconds;
    result->speculative_fallback_seconds = speculative_fallback_seconds;
    result->speculative_mtp_cache_clone_seconds =
        speculative_mtp_cache_clone_seconds;
    result->speculative_target_cache_clone_seconds =
        speculative_target_cache_clone_seconds;
    result->speculative_argmax_seconds = speculative_argmax_seconds;
    result->speculative_rounds = speculative_rounds;
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