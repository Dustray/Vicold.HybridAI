#include "models/qwen_reference.h"

#include "core/device_manager.h"
#include "ops/elementwise.h"
#include "ops/linear.h"
#include "ops/rmsnorm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

namespace hybridai {

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

Status lookup_embedding(const Tensor& embedding, const std::vector<int64_t>& ids,
                        Backend* backend, const Device& device,
                        Tensor* output) {
    return lookup_embedding_into(embedding, ids, backend, device, nullptr,
                                 nullptr, output);
}

Status lookup_embedding_into(
    const Tensor& embedding, const std::vector<int64_t>& ids,
    Backend* backend, const Device& device,
    const std::shared_ptr<Buffer>& output_buffer,
    const std::shared_ptr<Buffer>& ids_buffer, Tensor* output) {
    if (output == nullptr || backend == nullptr ||
        embedding.buffer() == nullptr || embedding.dtype() != DType::BF16 ||
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
    const size_t row_bytes = static_cast<size_t>(hidden_size) *
                             SizeOfDType(DType::BF16);
    std::shared_ptr<Buffer> buffer = output_buffer;
    if (buffer == nullptr || buffer->size() < ids.size() * row_bytes ||
        buffer->memory_type() != MemoryType::Device) {
        buffer = backend->create_buffer(ids.size() * row_bytes,
                                        MemoryType::Device);
    }
    if (buffer == nullptr) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate embedding output");
    }
    Status status;
    if (device.is_gpu()) {
        std::shared_ptr<Buffer> device_ids = ids_buffer;
        if (device_ids == nullptr ||
            device_ids->size() < ids.size() * sizeof(int64_t) ||
            device_ids->memory_type() != MemoryType::Device) {
            device_ids = backend->create_buffer(
                ids.size() * sizeof(int64_t), MemoryType::Device);
        }
        if (device_ids == nullptr) {
            return Status(StatusCode::OutOfMemory,
                          "Failed to allocate embedding ids");
        }
        status = backend->memcpy_h2d(device_ids.get(), ids.data(),
                                     ids.size() * sizeof(int64_t));
        if (!status.ok()) return status;
        status = backend->embedding_gather(
            buffer.get(), embedding.buffer().get(), device_ids.get(),
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
    *output = Tensor(Shape{static_cast<int64_t>(ids.size()), hidden_size},
                     DType::BF16, device, std::move(buffer));
    return Status::OK();
}

Tensor qwen_rmsnorm_reference(const Tensor& input, const Tensor& weight,
                              Backend* backend, const Device& device,
                              float eps) {
    if (backend == nullptr || input.dtype() != weight.dtype() ||
        input.buffer() == nullptr || weight.buffer() == nullptr ||
        input.shape().ndim() == 0 || weight.shape().ndim() != 1 ||
        weight.shape().dim(0) != input.shape().dim(input.shape().ndim() - 1) ||
        input.device() != device || weight.device() != device) {
        return Tensor();
    }
    return ops::RMSNorm::forward(input, weight, eps, nullptr, true);
}

Tensor make_residual(const Tensor& lhs, const Tensor& rhs, Backend* backend,
                     const Device& device) {
    if (backend == nullptr || lhs.dtype() != rhs.dtype() ||
        lhs.shape() != rhs.shape() || lhs.device() != device ||
        rhs.device() != device || lhs.buffer() == nullptr ||
        rhs.buffer() == nullptr) return Tensor();
    const size_t count = static_cast<size_t>(lhs.numel());
    auto buffer = backend->create_buffer(
        lhs.nbytes(), device.is_gpu() ? MemoryType::Device : MemoryType::Host);
    if (buffer == nullptr) return Tensor();
    if (device.is_gpu()) {
        if (!backend->add(buffer.get(), lhs.buffer().get(), rhs.buffer().get(),
                           lhs.dtype(), static_cast<int64_t>(count)).ok()) {
            return Tensor();
        }
    } else {
        if (lhs.dtype() != DType::FP32) return Tensor();
        const float* a = static_cast<const float*>(lhs.data());
        const float* b = static_cast<const float*>(rhs.data());
        float* out = static_cast<float*>(buffer->data());
        for (size_t i = 0; i < count; ++i) out[i] = a[i] + b[i];
    }
    return Tensor(lhs.shape(), lhs.dtype(), device, std::move(buffer));
}

Tensor qwen_mlp_reference(const Tensor& input,
                          const models::Qwen3LayerWeights& layer,
                          Backend* backend, const Device& device) {
    if (backend == nullptr || input.buffer() == nullptr ||
        input.device() != device ||
        input.dtype() != layer.mlp_gate_proj.values.dtype()) return Tensor();
    Status validation = ops::Linear::validate(
        input, layer.mlp_gate_proj.values, true, nullptr);
    if (!validation.ok()) return Tensor();
    validation = ops::Linear::validate(input, layer.mlp_up_proj.values, true,
                                       nullptr);
    if (!validation.ok()) return Tensor();
    Tensor gate = ops::Linear::forward(input, layer.mlp_gate_proj.values, true);
    Tensor up = ops::Linear::forward(input, layer.mlp_up_proj.values, true);
    if (gate.buffer() == nullptr || up.buffer() == nullptr) return Tensor();
    Tensor activated;
    if (input.device().is_gpu()) {
        if (!backend->silu_mul(gate.buffer().get(), gate.buffer().get(),
                               up.buffer().get(), gate.dtype(), gate.numel()).ok()) {
            return Tensor();
        }
        activated = std::move(gate);
    } else {
        activated = ops::Elementwise::silu_mul(gate, up);
    }
    if (activated.buffer() == nullptr ||
        !ops::Linear::validate(activated, layer.mlp_down_proj.values, true,
                               nullptr).ok()) return Tensor();
    return ops::Linear::forward(activated, layer.mlp_down_proj.values, true);
}

Tensor qwen_attention_reference(
    const Tensor& input, const models::Qwen3LayerWeights& layer,
    Backend* backend, const Device& device, const models::Qwen3Config& config,
    ops::AttentionKVCache* cache, int64_t max_cache_len) {
    (void)device;
    if (!layer.is_attention_layer || backend == nullptr) return Tensor();
    const Tensor& wq = layer.q_proj.values;
    const Tensor& wk = layer.k_proj.values;
    const Tensor& wv = layer.v_proj.values;
    const Tensor& wo = layer.o_proj.values;
    if (!ops::GatedGQAAttention::validate(
             input, wq, wk, wv, wo, config.num_attention_heads,
             config.num_key_value_heads, config.head_dim, config.rope_head_dim,
             config.rope_theta).ok()) return Tensor();
    if (cache != nullptr) {
        return ops::GatedGQAAttention::forward_cached(
            input, wq, wk, wv, wo, config.num_attention_heads,
            config.num_key_value_heads, config.head_dim, config.rope_head_dim,
            max_cache_len, cache, config.rope_theta, nullptr, layer.q_norm,
            layer.k_norm, config.rms_norm_eps);
    }
    return ops::GatedGQAAttention::forward(
        input, wq, wk, wv, wo, config.num_attention_heads,
        config.num_key_value_heads, config.head_dim, config.rope_head_dim,
        config.rope_theta, nullptr, layer.q_norm, layer.k_norm,
        config.rms_norm_eps);
}

Tensor qwen_deltanet_reference(
    const Tensor& input, const models::Qwen3LayerWeights& layer,
    Backend* backend, const Device& device, const models::Qwen3Config& config,
    ops::DeltaNetCache* cache) {
    ops::DeltaNetCache local_cache;
    ops::DeltaNetCache* active_cache = cache == nullptr ? &local_cache : cache;
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
    std::shared_ptr<Buffer> qkv_workspace;
    std::shared_ptr<Buffer> z_workspace;
    std::shared_ptr<Buffer> a_workspace;
    std::shared_ptr<Buffer> b_workspace;
    const auto project = [&](const Tensor& weight, std::shared_ptr<Buffer>* workspace) {
        const size_t bytes = static_cast<size_t>(tokens) *
                             static_cast<size_t>(weight.shape().dim(0)) *
                             SizeOfDType(input.dtype());
        if (*workspace == nullptr || (*workspace)->size() < bytes) {
            *workspace = backend->create_buffer(bytes, input.buffer()->memory_type());
        }
        return *workspace == nullptr
                   ? Tensor()
                   : ops::Linear::forward_into(input, weight, *workspace, true);
    };
    Tensor qkv_projection = project(layer.in_proj_qkv.values, &qkv_workspace);
    Tensor z = project(layer.in_proj_z.values, &z_workspace);
    Tensor a = project(layer.in_proj_a, &a_workspace);
    Tensor b = project(layer.in_proj_b, &b_workspace);
    if (qkv_projection.buffer() == nullptr || z.buffer() == nullptr ||
        a.buffer() == nullptr || b.buffer() == nullptr) return Tensor();
    ops::DeltaNetQKV qkv = ops::GatedDeltaNet::grouped_causal_conv(
        qkv_projection, layer.conv1d_weight, qk_heads, value_heads, key_dim,
        value_dim, config.linear_conv_kernel_dim, active_cache);
    if (!qkv.valid()) return Tensor();
    Tensor recurrent = ops::GatedDeltaNet::recurrent(
        qkv.query, qkv.key, qkv.value,
        a.reshape(Shape{tokens, value_heads}),
        b.reshape(Shape{tokens, value_heads}),
        z.reshape(Shape{tokens, value_heads, value_dim}), layer.a_log,
        layer.dt_bias, layer.linear_attn_norm, qk_heads, value_heads, key_dim,
        value_dim, config.rms_norm_eps, active_cache);
    if (recurrent.buffer() == nullptr) return Tensor();
    return ops::Linear::forward(
        recurrent.reshape(Shape{tokens, value_heads * value_dim}),
        layer.linear_out_proj.values, true);
}

}  // namespace hybridai
