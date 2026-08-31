#include "ops/delta_net.h"

#include "backends/backend_registry.h"
#include "ops/linear.h"
#include "ops/registry.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace hybridai {
namespace ops {

Status GatedDeltaNet::validate(const Tensor& input,
                              const Tensor& wq, const Tensor& wk,
                              const Tensor& wv, const Tensor& wo,
                              int64_t num_qk_heads, int64_t num_v_heads,
                              int64_t head_dim) {
    if (input.dtype() != DType::FP32) {
        return Status(StatusCode::UnsupportedDType,
                        "GatedDeltaNet CPU only supports FP32");
    }
    if (input.shape().ndim() < 2) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet input must be at least 2-D");
    }
    const int64_t hidden_size = input.shape().dim(input.shape().ndim() - 1);
    if (hidden_size <= 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet hidden_size must be positive");
    }
    if (num_qk_heads <= 0 || num_v_heads <= 0 || head_dim <= 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet head counts/dims must be positive");
    }
    if (num_v_heads % num_qk_heads != 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet num_v_heads must be divisible by num_qk_heads");
    }

    const int64_t qk_out = num_qk_heads * head_dim;
    const int64_t v_out = num_v_heads * head_dim;

    Shape unused;
    Status s = Linear::compute_output_shape(input.shape(), wq.shape(), true, &unused);
    if (!s.ok()) return s;
    if (wq.shape().dim(0) != qk_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet wq output dim mismatch");
    }
    if (wk.shape().dim(0) != qk_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet wk output dim mismatch");
    }
    if (wv.shape().dim(0) != v_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet wv output dim mismatch");
    }
    if (wo.shape().dim(0) != hidden_size) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet wo output dim mismatch");
    }
    if (wo.shape().dim(1) != v_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedDeltaNet wo input dim mismatch");
    }

    return Status::OK();
}

Tensor GatedDeltaNet::forward(const Tensor& input,
                               const Tensor& wq, const Tensor& wk,
                               const Tensor& wv, const Tensor& wo,
                               int64_t num_qk_heads, int64_t num_v_heads,
                               int64_t head_dim, Stream* stream) {
    Status status = validate(input, wq, wk, wv, wo,
                             num_qk_heads, num_v_heads, head_dim);
    if (!status.ok()) {
        return Tensor();
    }

    const size_t ndim = input.shape().ndim();
    int64_t seq_len = 1;
    for (size_t i = 0; i + 1 < ndim; ++i) {
        seq_len *= input.shape().dim(i);
    }

    // 1. Project q/k/v.
    Tensor q = Linear::forward(input, wq, true, nullptr, stream);
    Tensor k = Linear::forward(input, wk, true, nullptr, stream);
    Tensor v = Linear::forward(input, wv, true, nullptr, stream);
    if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr) {
        return Tensor();
    }

    const float* q_ptr = static_cast<const float*>(q.data());
    const float* k_ptr = static_cast<const float*>(k.data());
    const float* v_ptr = static_cast<const float*>(v.data());

    const int64_t v_per_qk = num_v_heads / num_qk_heads;

    // 2. Recurrent linear attention with per-QK-head gate.
    // State shape per QK head: [head_dim, head_dim]. We keep one state per QK
    // head and share it across the v heads that belong to that QK head.
    auto backend = BackendRegistry::instance().get_backend(input.device());
    if (backend == nullptr) {
        return Tensor();
    }

    const size_t out_bytes =
        static_cast<size_t>(seq_len * num_v_heads * head_dim) * sizeof(float);
    auto out_buf = backend->create_buffer(out_bytes,
                                        input.buffer()->memory_type());
    if (out_buf == nullptr) {
        return Tensor();
    }
    float* out_ptr = static_cast<float*>(out_buf->data());
    std::memset(out_ptr, 0, out_bytes);

    std::vector<std::vector<float>> states(num_qk_heads,
        std::vector<float>(head_dim * head_dim, 0.0f));

    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t qkh = 0; qkh < num_qk_heads; ++qkh) {
            const float* q_head = q_ptr + (t * num_qk_heads + qkh) * head_dim;
            const float* k_head = k_ptr + (t * num_qk_heads + qkh) * head_dim;

            // Decay factor per QK head (deterministic placeholder).
            float gate = 0.99f;

            // Update state: S = gate * S + k^T * v for each v head in group.
            for (int64_t vh = 0; vh < v_per_qk; ++vh) {
                const int64_t v_head_idx = qkh * v_per_qk + vh;
                const float* v_head =
                    v_ptr + (t * num_v_heads + v_head_idx) * head_dim;
                float* state = states[qkh].data();

                for (int64_t i = 0; i < head_dim; ++i) {
                    for (int64_t j = 0; j < head_dim; ++j) {
                        state[i * head_dim + j] =
                            gate * state[i * head_dim + j] +
                            k_head[i] * v_head[j];
                    }
                }

                // Output: o = q @ S
                float* o_head =
                    out_ptr + (t * num_v_heads + v_head_idx) * head_dim;
                for (int64_t j = 0; j < head_dim; ++j) {
                    o_head[j] = 0.0f;
                }
                for (int64_t i = 0; i < head_dim; ++i) {
                    float qv = q_head[i];
                    for (int64_t j = 0; j < head_dim; ++j) {
                        o_head[j] += qv * state[i * head_dim + j];
                    }
                }
            }
        }
    }

    Tensor output_2d(
        Shape{seq_len, num_v_heads * head_dim}, DType::FP32,
        input.device(), out_buf);

    // 3. Output projection.
    return Linear::forward(output_2d, wo, true, nullptr, stream);
}

Tensor GatedDeltaNet::causal_conv(const Tensor& input, const Tensor& weight,
                                  int64_t kernel_size, DeltaNetCache* cache,
                                  Stream* stream) {
    if (cache == nullptr || input.buffer() == nullptr ||
        weight.buffer() == nullptr || input.device().type() == DeviceType::CPU ||
        input.shape().ndim() != 2 || weight.shape().numel() !=
            input.shape().dim(1) * kernel_size ||
        input.dtype() != weight.dtype() || kernel_size <= 1) {
        return Tensor();
    }
    auto backend = BackendRegistry::instance().get_backend(input.device());
    if (backend == nullptr) return Tensor();
    const int64_t token_count = input.shape().dim(0);
    const int64_t channels = input.shape().dim(1);
    const Shape state_shape{channels, kernel_size - 1};
    if (cache->conv_state.buffer() == nullptr) {
        auto state_buffer = backend->create_buffer(
            static_cast<size_t>(state_shape.numel()) * SizeOfDType(input.dtype()),
            input.buffer()->memory_type());
        if (state_buffer == nullptr ||
            !backend->memset(state_buffer.get(), 0, state_buffer->size()).ok())
            return Tensor();
        cache->conv_state = Tensor(state_shape, input.dtype(), input.device(),
                                   std::move(state_buffer));
    } else if (cache->conv_state.shape() != state_shape ||
               cache->conv_state.dtype() != input.dtype() ||
               cache->conv_state.device() != input.device()) {
        return Tensor();
    }
    auto output_buffer = backend->create_buffer(
        input.nbytes(), input.buffer()->memory_type());
    if (output_buffer == nullptr ||
        !backend->causal_conv1d_silu(
            output_buffer.get(), cache->conv_state.buffer().get(),
            input.buffer().get(), weight.buffer().get(), input.dtype(),
            token_count, channels, kernel_size, stream).ok()) return Tensor();
    return Tensor(input.shape(), input.dtype(), input.device(),
                  std::move(output_buffer));
}

DeltaNetQKV GatedDeltaNet::grouped_causal_conv(
    const Tensor& grouped_qkv, const Tensor& weight,
    int64_t num_qk_heads, int64_t num_value_heads,
    int64_t key_head_dim, int64_t value_head_dim, int64_t kernel_size,
    DeltaNetCache* cache, Stream* stream) {
    DeltaNetQKV output;
    const int64_t key_width = num_qk_heads * key_head_dim;
    const int64_t value_width = num_value_heads * value_head_dim;
    const int64_t channels = key_width * 2 + value_width;
    if (cache == nullptr || grouped_qkv.buffer() == nullptr ||
        weight.buffer() == nullptr || grouped_qkv.device().type() == DeviceType::CPU ||
        grouped_qkv.shape().ndim() != 2 ||
        grouped_qkv.shape().dim(1) != channels ||
        weight.numel() != channels * kernel_size ||
        grouped_qkv.dtype() != weight.dtype() || num_qk_heads <= 0 ||
        num_value_heads % num_qk_heads != 0 || kernel_size <= 1) return output;
    auto backend = BackendRegistry::instance().get_backend(
        grouped_qkv.device());
    if (backend == nullptr) return output;
    const int64_t token_count = grouped_qkv.shape().dim(0);
    const Shape state_shape{channels, kernel_size - 1};
    if (cache->conv_state.buffer() == nullptr) {
        auto state_buffer = backend->create_buffer(
            static_cast<size_t>(state_shape.numel()) *
                SizeOfDType(grouped_qkv.dtype()),
            grouped_qkv.buffer()->memory_type());
        if (state_buffer == nullptr ||
            !backend->memset(state_buffer.get(), 0, state_buffer->size()).ok())
            return output;
        cache->conv_state = Tensor(state_shape, grouped_qkv.dtype(),
                                   grouped_qkv.device(),
                                   std::move(state_buffer));
    } else if (cache->conv_state.shape() != state_shape ||
               cache->conv_state.dtype() != grouped_qkv.dtype() ||
               cache->conv_state.device() != grouped_qkv.device()) return output;
    auto q_buffer = backend->create_buffer(
        static_cast<size_t>(token_count * key_width) *
            SizeOfDType(grouped_qkv.dtype()),
        grouped_qkv.buffer()->memory_type());
    auto k_buffer = backend->create_buffer(
        static_cast<size_t>(token_count * key_width) *
            SizeOfDType(grouped_qkv.dtype()),
        grouped_qkv.buffer()->memory_type());
    auto v_buffer = backend->create_buffer(
        static_cast<size_t>(token_count * value_width) *
            SizeOfDType(grouped_qkv.dtype()),
        grouped_qkv.buffer()->memory_type());
    if (q_buffer == nullptr || k_buffer == nullptr || v_buffer == nullptr ||
        !backend->deltanet_grouped_conv(
            q_buffer.get(), k_buffer.get(), v_buffer.get(),
            cache->conv_state.buffer().get(), grouped_qkv.buffer().get(),
            weight.buffer().get(), grouped_qkv.dtype(), token_count,
            num_qk_heads, num_value_heads, key_head_dim, value_head_dim,
            kernel_size, stream).ok()) return output;
    output.query = Tensor(Shape{token_count, num_qk_heads, key_head_dim},
                          grouped_qkv.dtype(), grouped_qkv.device(),
                          std::move(q_buffer));
    output.key = Tensor(Shape{token_count, num_qk_heads, key_head_dim},
                        grouped_qkv.dtype(), grouped_qkv.device(),
                        std::move(k_buffer));
    output.value = Tensor(Shape{token_count, num_value_heads, value_head_dim},
                          grouped_qkv.dtype(), grouped_qkv.device(),
                          std::move(v_buffer));
    return output;
}

Tensor GatedDeltaNet::recurrent(
    const Tensor& query, const Tensor& key, const Tensor& value,
    const Tensor& a, const Tensor& b, const Tensor& z, const Tensor& a_log,
    const Tensor& dt_bias, const Tensor& norm_weight, int64_t num_qk_heads,
    int64_t num_value_heads, int64_t key_head_dim, int64_t value_head_dim,
    float eps, DeltaNetCache* cache, Stream* stream) {
    if (cache == nullptr || query.buffer() == nullptr || key.buffer() == nullptr ||
        value.buffer() == nullptr || a.buffer() == nullptr || b.buffer() == nullptr ||
        z.buffer() == nullptr || a_log.buffer() == nullptr ||
        dt_bias.buffer() == nullptr || norm_weight.buffer() == nullptr ||
        query.device().type() == DeviceType::CPU || query.shape().ndim() != 3 ||
        query.dtype() != key.dtype() || query.dtype() != value.dtype() ||
        query.dtype() != a.dtype() || query.dtype() != b.dtype() ||
        query.dtype() != z.dtype() || query.dtype() != a_log.dtype() ||
        query.dtype() != dt_bias.dtype() || query.dtype() != norm_weight.dtype())
        return Tensor();
    const int64_t token_count = query.shape().dim(0);
    if (query.shape() != Shape{token_count, num_qk_heads, key_head_dim} ||
        key.shape() != query.shape() ||
        value.shape() != Shape{token_count, num_value_heads, value_head_dim} ||
        z.shape() != value.shape() ||
        a.numel() != token_count * num_value_heads ||
        b.numel() != token_count * num_value_heads ||
        a_log.numel() != num_value_heads || dt_bias.numel() != num_value_heads ||
        norm_weight.numel() != value_head_dim ||
        num_value_heads % num_qk_heads != 0 || eps <= 0.0f) return Tensor();
    auto backend = BackendRegistry::instance().get_backend(query.device());
    if (backend == nullptr) return Tensor();
    const Shape state_shape{num_value_heads, key_head_dim, value_head_dim};
    if (cache->recurrent_state.buffer() == nullptr) {
        auto state_buffer = backend->create_buffer(
            static_cast<size_t>(state_shape.numel()) * sizeof(float),
            query.buffer()->memory_type());
        if (state_buffer == nullptr ||
            !backend->memset(state_buffer.get(), 0, state_buffer->size()).ok())
            return Tensor();
        cache->recurrent_state = Tensor(state_shape, DType::FP32,
                                        query.device(), std::move(state_buffer));
    } else if (cache->recurrent_state.shape() != state_shape ||
               cache->recurrent_state.dtype() != DType::FP32 ||
               cache->recurrent_state.device() != query.device()) return Tensor();
    auto output_buffer = backend->create_buffer(
        value.nbytes(), value.buffer()->memory_type());
    if (output_buffer == nullptr || !backend->deltanet_recurrent(
            output_buffer.get(), cache->recurrent_state.buffer().get(),
            query.buffer().get(), key.buffer().get(), value.buffer().get(),
            a.buffer().get(), b.buffer().get(), a_log.buffer().get(),
            dt_bias.buffer().get(), norm_weight.buffer().get(), z.buffer().get(),
            query.dtype(), token_count, num_qk_heads, num_value_heads,
            key_head_dim, value_head_dim, eps, stream).ok()) return Tensor();
    cache->length += token_count;
    return Tensor(value.shape(), value.dtype(), value.device(),
                  std::move(output_buffer));
}

} // namespace ops
} // namespace hybridai
