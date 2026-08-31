#include "ops/attention.h"

#include "backends/backend_registry.h"
#include "ops/linear.h"
#include "ops/registry.h"
#include "ops/rmsnorm.h"
#include "ops/rope.h"
#include "ops/softmax.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <vector>

namespace hybridai {
namespace ops {

namespace {

Tensor convert_dtype_host(const Tensor& input, DType output_dtype,
                          Backend* backend) {
    if (backend == nullptr || input.buffer() == nullptr) return Tensor();
    if (input.dtype() == output_dtype) return input;
    const size_t count = static_cast<size_t>(input.numel());
    if (input.device().type() != DeviceType::CPU) {
        auto buffer = backend->create_buffer(
            count * SizeOfDType(output_dtype), input.buffer()->memory_type());
        if (buffer == nullptr ||
            !backend->cast(buffer.get(), input.buffer().get(), output_dtype,
                           input.dtype(), static_cast<int64_t>(count)).ok()) {
            return Tensor();
        }
        return Tensor(input.shape(), output_dtype, input.device(),
                      std::move(buffer));
    }
    std::vector<uint8_t> source(input.nbytes());
    if (!backend->memcpy_d2h(source.data(), input.buffer().get(),
                             input.nbytes()).ok()) {
        return Tensor();
    }
    std::vector<uint8_t> converted(count * SizeOfDType(output_dtype));
    if (input.dtype() == DType::FP32 && output_dtype == DType::BF16) {
        const float* src = reinterpret_cast<const float*>(source.data());
        auto* dst = reinterpret_cast<uint16_t*>(converted.data());
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = 0;
            std::memcpy(&bits, src + i, sizeof(bits));
            dst[i] = static_cast<uint16_t>(bits >> 16);
        }
    } else if (input.dtype() == DType::BF16 && output_dtype == DType::FP32) {
        const auto* src = reinterpret_cast<const uint16_t*>(source.data());
        auto* dst = reinterpret_cast<float*>(converted.data());
        for (size_t i = 0; i < count; ++i) {
            uint32_t bits = static_cast<uint32_t>(src[i]) << 16;
            std::memcpy(dst + i, &bits, sizeof(bits));
        }
    } else {
        return Tensor();
    }
    auto buffer = backend->create_buffer(converted.size(), MemoryType::Unified);
    if (buffer == nullptr ||
        !backend->memcpy_h2d(buffer.get(), converted.data(), converted.size()).ok()) {
        return Tensor();
    }
    return Tensor(input.shape(), output_dtype, input.device(), std::move(buffer));
}

Tensor apply_partial_rope(const Tensor& input, int64_t rope_head_dim,
                          int64_t position_offset, float base,
                          Backend* backend) {
    if (backend == nullptr || input.buffer() == nullptr ||
    (input.dtype() != DType::FP32 &&
     !(input.device().is_gpu() && input.dtype() == DType::BF16)) ||
    input.shape().ndim() != 3 ||
        rope_head_dim <= 0 || rope_head_dim > input.shape().dim(2) ||
        (rope_head_dim % 2) != 0) {
        return Tensor();
    }
    if (input.device().type() != DeviceType::CPU) {
        return RoPE::partial(input, rope_head_dim, position_offset, base);
    }
    const int64_t seq_len = input.shape().dim(0);
    const int64_t num_heads = input.shape().dim(1);
    const int64_t head_dim = input.shape().dim(2);
    std::vector<float> host(static_cast<size_t>(input.numel()));
    if (!backend->memcpy_d2h(host.data(), input.buffer().get(), input.nbytes()).ok()) {
        return Tensor();
    }
    for (int64_t s = 0; s < seq_len; ++s) {
        for (int64_t h = 0; h < num_heads; ++h) {
            float* head = host.data() + (s * num_heads + h) * head_dim;
            // Qwen3.5 follows Transformers' rotate_half convention: the
            // rotary part is laid out as [x_1...x_n, y_1...y_n], not as
            // adjacent complex pairs.  cos/sin are duplicated across the
            // rotary dimension and rotate_half(x) = [-y, x].
            const int64_t half = rope_head_dim / 2;
            std::vector<float> rotary(static_cast<size_t>(rope_head_dim));
            std::memcpy(rotary.data(), head,
                        static_cast<size_t>(rope_head_dim) * sizeof(float));
            for (int64_t j = 0; j < half; ++j) {
                const float theta = static_cast<float>(position_offset + s) /
                    std::pow(base, 2.0f * static_cast<float>(j) /
                                       static_cast<float>(rope_head_dim));
                const float c = std::cos(theta);
                const float sn = std::sin(theta);
                const float x = rotary[static_cast<size_t>(j)];
                const float y = rotary[static_cast<size_t>(j + half)];
                head[j] = x * c - y * sn;
                head[j + half] = y * c + x * sn;
            }
        }
    }
    auto buffer = backend->create_buffer(input.nbytes(), MemoryType::Unified);
    if (buffer == nullptr ||
        !backend->memcpy_h2d(buffer.get(), host.data(), input.nbytes()).ok()) {
        return Tensor();
    }
    return Tensor(input.shape(), DType::FP32, input.device(), std::move(buffer));
}

} // namespace

Status GatedGQAAttention::validate(const Tensor& input,
                                    const Tensor& wq, const Tensor& wk,
                                    const Tensor& wv, const Tensor& wo,
                                    int64_t num_q_heads, int64_t num_kv_heads,
                                    int64_t head_dim, int64_t rope_head_dim,
                                    float rope_base) {
    (void)rope_base;
    if (input.dtype() != DType::FP32 &&
        !(input.device().is_gpu() && input.dtype() == DType::BF16)) {
        return Status(StatusCode::UnsupportedDType,
                        "GatedGQAAttention supports CPU FP32 and GPU FP32/BF16 activations");
    }
    if (input.shape().ndim() < 2) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention input must be at least 2-D");
    }
    const int64_t hidden_size = input.shape().dim(input.shape().ndim() - 1);
    if (hidden_size <= 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention hidden_size must be positive");
    }
    if (num_q_heads <= 0 || num_kv_heads <= 0 || head_dim <= 0 ||
        rope_head_dim <= 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention head counts/dims must be positive");
    }
    if (num_q_heads % num_kv_heads != 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention num_q_heads must be divisible by num_kv_heads");
    }
    if (rope_head_dim % 2 != 0) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention rope_head_dim must be even");
    }
    if (head_dim < rope_head_dim) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention head_dim must be >= rope_head_dim");
    }

    // Weight shapes: [out_features, in_features], Linear uses transpose_weight=true.
    // Wq out = num_q_heads * head_dim
    // Wk/Wv out = num_kv_heads * head_dim
    // Wo out = hidden_size
    const int64_t q_out = num_q_heads * head_dim;
    const int64_t kv_out = num_kv_heads * head_dim;

    Shape unused;
    Status s = Linear::compute_output_shape(input.shape(), wq.shape(), true, &unused);
    if (!s.ok()) return s;
    // Qwen3.5/Qwen3.8 gated attention stores query and gate in one
    // projection: [query, gate], hence q_proj may have 2 * q_out rows.
    if (wq.shape().dim(0) != q_out && wq.shape().dim(0) != 2 * q_out) {
        return Status(StatusCode::InvalidArgument,
                      "GatedGQAAttention wq output dim mismatch");
    }
    if (wk.shape().dim(0) != kv_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention wk output dim mismatch");
    }
    if (wv.shape().dim(0) != kv_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention wv output dim mismatch");
    }
    if (wo.shape().dim(0) != hidden_size) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention wo output dim mismatch");
    }
    if (wo.shape().dim(1) != q_out) {
        return Status(StatusCode::InvalidArgument,
                        "GatedGQAAttention wo input dim mismatch");
    }

    return Status::OK();
}

Tensor GatedGQAAttention::forward(const Tensor& input,
                                 const Tensor& wq, const Tensor& wk,
                                 const Tensor& wv, const Tensor& wo,
                                 int64_t num_q_heads, int64_t num_kv_heads,
                                 int64_t head_dim, int64_t rope_head_dim,
                                 float rope_base, Stream* stream,
                                 const Tensor& q_norm_weight,
                                 const Tensor& k_norm_weight,
                                 float rms_norm_eps) {
    return forward_impl(input, wq, wk, wv, wo, num_q_heads, num_kv_heads,
                        head_dim, rope_head_dim, rope_base, stream,
                        q_norm_weight, k_norm_weight, rms_norm_eps, 0, nullptr);
}

Tensor GatedGQAAttention::forward_cached(
    const Tensor& input, const Tensor& wq, const Tensor& wk,
    const Tensor& wv, const Tensor& wo, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rope_head_dim,
    int64_t max_seq_len, AttentionKVCache* cache, float rope_base,
    Stream* stream, const Tensor& q_norm_weight,
    const Tensor& k_norm_weight, float rms_norm_eps) {
    return forward_impl(input, wq, wk, wv, wo, num_q_heads, num_kv_heads,
                        head_dim, rope_head_dim, rope_base, stream,
                        q_norm_weight, k_norm_weight, rms_norm_eps,
                        max_seq_len, cache);
}

Tensor GatedGQAAttention::forward_impl(
    const Tensor& input, const Tensor& wq, const Tensor& wk,
    const Tensor& wv, const Tensor& wo, int64_t num_q_heads,
    int64_t num_kv_heads, int64_t head_dim, int64_t rope_head_dim,
    float rope_base, Stream* stream, const Tensor& q_norm_weight,
    const Tensor& k_norm_weight, float rms_norm_eps, int64_t max_seq_len,
    AttentionKVCache* cache) {
    Status status = validate(input, wq, wk, wv, wo,
                             num_q_heads, num_kv_heads,
                             head_dim, rope_head_dim, rope_base);
    if (!status.ok()) {
        return Tensor();
    }

    const size_t ndim = input.shape().ndim();
    int64_t seq_len = 1;
    for (size_t i = 0; i + 1 < ndim; ++i) {
        seq_len *= input.shape().dim(i);
    }

    auto backend = BackendRegistry::instance().get_backend(input.device());
    if (backend == nullptr) {
        return Tensor();
    }
    const bool use_cache = cache != nullptr;
    if (use_cache && (input.device().type() == DeviceType::CPU ||
                      max_seq_len <= 0 || cache->length < 0 ||
                      cache->length + seq_len > max_seq_len)) {
        return Tensor();
    }
    const bool gpu_native = input.device().is_gpu();
    const bool mixed_bf16 = !gpu_native && wq.dtype() == DType::BF16;
    const DType activation_dtype = input.dtype();
    Tensor projection_input = mixed_bf16
        ? convert_dtype_host(input, DType::BF16, backend.get()) : input;
    if (projection_input.buffer() == nullptr) {
        return Tensor();
    }

    // 1. Project Q/K/V.
    Tensor q = Linear::forward(projection_input, wq, true, nullptr, stream);
    Tensor k = Linear::forward(projection_input, wk, true, nullptr, stream);
    Tensor v = Linear::forward(projection_input, wv, true, nullptr, stream);
    if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr) {
        return Tensor();
    }
    if (!gpu_native) {
        q = convert_dtype_host(q, DType::FP32, backend.get());
        k = convert_dtype_host(k, DType::FP32, backend.get());
        v = convert_dtype_host(v, DType::FP32, backend.get());
    }
    if (q.buffer() == nullptr || k.buffer() == nullptr || v.buffer() == nullptr) {
        return Tensor();
    }

    // Reshape to [seq_len, num_heads, head_dim] and apply RoPE.
    const bool has_gate = wq.shape().dim(0) == 2 * num_q_heads * head_dim;
    Tensor q_query = q;
    Tensor q_gate;
    if (has_gate) {
        const int64_t query_numel = seq_len * num_q_heads * head_dim;
        auto query_buf = backend->create_buffer(
            static_cast<size_t>(query_numel) * SizeOfDType(activation_dtype),
            input.buffer()->memory_type());
        auto gate_buf = backend->create_buffer(
            static_cast<size_t>(query_numel) * SizeOfDType(activation_dtype),
            input.buffer()->memory_type());
        if (query_buf == nullptr || gate_buf == nullptr) return Tensor();
        if (input.device().type() != DeviceType::CPU) {
            if (!backend->split_q_gate(
                    query_buf.get(), gate_buf.get(), q.buffer().get(),
                    activation_dtype, seq_len, num_q_heads, head_dim,
                    stream).ok()) {
                return Tensor();
            }
        } else {
            const float* q_all = static_cast<const float*>(q.data());
            float* query_ptr = static_cast<float*>(query_buf->data());
            float* gate_ptr = static_cast<float*>(gate_buf->data());
            for (int64_t t = 0; t < seq_len; ++t) {
                for (int64_t h = 0; h < num_q_heads; ++h) {
                    const float* head = q_all +
                        (t * num_q_heads + h) * (2 * head_dim);
                    float* query_head = query_ptr +
                        (t * num_q_heads + h) * head_dim;
                    float* gate_head = gate_ptr +
                        (t * num_q_heads + h) * head_dim;
                    std::memcpy(query_head, head,
                                static_cast<size_t>(head_dim) * sizeof(float));
                    std::memcpy(gate_head, head + head_dim,
                                static_cast<size_t>(head_dim) * sizeof(float));
                }
            }
        }
        q_query = Tensor(Shape{seq_len, num_q_heads * head_dim},
                 activation_dtype, input.device(),
                 std::move(query_buf));
        q_gate = Tensor(Shape{seq_len, num_q_heads * head_dim},
                activation_dtype, input.device(),
                std::move(gate_buf));
    }
    Tensor q_3d(q_query.reshape(Shape{seq_len, num_q_heads, head_dim}));
    Tensor k_3d(k.reshape(Shape{seq_len, num_kv_heads, head_dim}));

    // Qwen3.5 applies a per-head RMSNorm to Q and K before RoPE. GPU tensors
    // keep their native dtype; the RMSNorm kernel performs FP32 accumulation
    // internally and writes the result back in the original dtype.
    if (input.device().type() != DeviceType::CPU &&
        q_norm_weight.buffer() != nullptr &&
        k_norm_weight.buffer() != nullptr) {
        if (q_norm_weight.dtype() != activation_dtype ||
            k_norm_weight.dtype() != activation_dtype)
            return Tensor();
        q_3d = RMSNorm::forward(
            q_3d, q_norm_weight, rms_norm_eps, stream, true);
        k_3d = RMSNorm::forward(
            k_3d, k_norm_weight, rms_norm_eps, stream, true);
        if (q_3d.buffer() == nullptr || k_3d.buffer() == nullptr) return Tensor();
    } else {
    std::vector<float> q_norm_host(static_cast<size_t>(head_dim), 1.0f);
    std::vector<float> k_norm_host(static_cast<size_t>(head_dim), 1.0f);
    auto read_norm = [&](const Tensor& weight, std::vector<float>* dst) {
        if (weight.buffer() == nullptr || weight.shape().ndim() != 1 ||
            weight.shape().dim(0) != head_dim) return true;
        Tensor weight_fp32 = convert_dtype_host(weight, DType::FP32, backend.get());
        if (weight_fp32.buffer() == nullptr) return false;
        return backend->memcpy_d2h(dst->data(), weight_fp32.buffer().get(),
                                   weight_fp32.nbytes()).ok();
    };
    if (!read_norm(q_norm_weight, &q_norm_host) ||
        !read_norm(k_norm_weight, &k_norm_host)) return Tensor();
    auto apply_head_rms = [&](Tensor* tensor, int64_t heads,
                              const std::vector<float>& weight) {
        std::vector<float> host(static_cast<size_t>(tensor->numel()));
        if (!backend->memcpy_d2h(host.data(), tensor->buffer().get(),
                                 tensor->nbytes()).ok()) return false;
        for (int64_t t = 0; t < seq_len; ++t) {
            for (int64_t h = 0; h < heads; ++h) {
                float* x = host.data() + (t * heads + h) * head_dim;
                float mean_sq = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) mean_sq += x[d] * x[d];
                const float inv_rms =
                    1.0f / std::sqrt(mean_sq / head_dim + rms_norm_eps);
                // Qwen3.5 uses the same residual scale parameterization as
                // its main RMSNorm: normalized * (1 + weight).
                for (int64_t d = 0; d < head_dim; ++d) {
                    x[d] *= inv_rms * (1.0f + weight[d]);
                }
            }
        }
        auto buffer = backend->create_buffer(tensor->nbytes(), MemoryType::Unified);
        if (buffer == nullptr || !backend->memcpy_h2d(buffer.get(), host.data(),
                                                        tensor->nbytes()).ok()) return false;
        *tensor = Tensor(tensor->shape(), DType::FP32, tensor->device(), std::move(buffer));
        return true;
    };
    if (!apply_head_rms(&q_3d, num_q_heads, q_norm_host) ||
        !apply_head_rms(&k_3d, num_kv_heads, k_norm_host)) return Tensor();
    }

    const int64_t position_offset = use_cache ? cache->length : 0;
    Tensor q_rot = apply_partial_rope(q_3d, rope_head_dim, position_offset,
                                      rope_base,
                                      backend.get());
    Tensor k_rot = apply_partial_rope(k_3d, rope_head_dim, position_offset,
                                      rope_base,
                                      backend.get());
    if (q_rot.data() == nullptr || k_rot.data() == nullptr) {
        return Tensor();
    }

    // 2. Compute attention scores and output.
    // For GQA each KV head is shared by q_heads_per_kv_head query heads.
    const int64_t q_heads_per_kv = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Buffer for concatenated attention output [seq_len, num_q_heads, head_dim].
    const size_t attn_out_bytes =
        static_cast<size_t>(seq_len * num_q_heads * head_dim) *
        SizeOfDType(activation_dtype);
    auto attn_out_buf = backend->create_buffer(attn_out_bytes,
                                               input.buffer()->memory_type());
    if (attn_out_buf == nullptr) {
        return Tensor();
    }
    if (input.device().type() != DeviceType::CPU) {
        Status attention_status;
        if (use_cache) {
            const Shape cache_shape{max_seq_len, num_kv_heads, head_dim};
            const size_t cache_bytes = static_cast<size_t>(
                max_seq_len * num_kv_heads * head_dim) *
                SizeOfDType(activation_dtype);
            if (!cache->initialized()) {
                auto key_cache = backend->create_buffer(
                    cache_bytes, input.buffer()->memory_type());
                auto value_cache = backend->create_buffer(
                    cache_bytes, input.buffer()->memory_type());
                if (key_cache == nullptr || value_cache == nullptr)
                    return Tensor();
                cache->key = Tensor(cache_shape, activation_dtype, input.device(),
                                    std::move(key_cache));
                cache->value = Tensor(cache_shape, activation_dtype, input.device(),
                                      std::move(value_cache));
                cache->capacity = max_seq_len;
            } else if (cache->capacity != max_seq_len ||
                       cache->key.shape() != cache_shape ||
                       cache->value.shape() != cache_shape ||
                       cache->key.dtype() != activation_dtype ||
                       cache->value.dtype() != activation_dtype ||
                       cache->key.device() != input.device() ||
                       cache->value.device() != input.device()) {
                return Tensor();
            }
            Status append_status = backend->append_kv_cache(
                cache->key.buffer().get(), cache->value.buffer().get(),
                k_rot.buffer().get(), v.buffer().get(), activation_dtype,
                seq_len, num_kv_heads, head_dim, cache->length,
                cache->capacity, stream);
            if (!append_status.ok()) return Tensor();
            const int64_t new_cache_len = cache->length + seq_len;
            attention_status = backend->cached_gqa(
                attn_out_buf.get(), q_rot.buffer().get(),
                cache->key.buffer().get(), cache->value.buffer().get(),
                has_gate ? q_gate.buffer().get() : nullptr, activation_dtype,
                seq_len, new_cache_len, num_q_heads, num_kv_heads, head_dim,
                stream);
            if (attention_status.ok()) cache->length = new_cache_len;
        } else {
            attention_status = backend->causal_gqa(
                attn_out_buf.get(), q_rot.buffer().get(), k_rot.buffer().get(),
                v.buffer().get(), has_gate ? q_gate.buffer().get() : nullptr,
                activation_dtype, seq_len, num_q_heads, num_kv_heads, head_dim,
                stream);
        }
        if (!attention_status.ok()) return Tensor();
        Tensor attn_out_tensor(
            Shape{seq_len, num_q_heads * head_dim}, activation_dtype,
            input.device(), attn_out_buf);
        return Linear::forward(attn_out_tensor, wo, true, nullptr, stream);
    }
    float* attn_out = static_cast<float*>(attn_out_buf->data());
    std::memset(attn_out, 0, attn_out_bytes);

    const float* q_ptr = static_cast<const float*>(q_rot.data());
    const float* k_ptr = static_cast<const float*>(k_rot.data());
    const float* v_ptr = static_cast<const float*>(v.data());

    // Per-row score buffer (causal mask).
    std::vector<float> scores(seq_len);
    std::vector<float> weights(seq_len);

    for (int64_t t = 0; t < seq_len; ++t) {
        for (int64_t qh = 0; qh < num_q_heads; ++qh) {
            const int64_t kvh = qh / q_heads_per_kv;
            const float* q_head = q_ptr + (t * num_q_heads + qh) * head_dim;
            float* out_head = attn_out + (t * num_q_heads + qh) * head_dim;

            // Compute scores against all previous positions (causal).
            for (int64_t s = 0; s <= t; ++s) {
                const float* k_head =
                    k_ptr + (s * num_kv_heads + kvh) * head_dim;
                float dot = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) {
                    dot += q_head[d] * k_head[d];
                }
                scores[s] = dot * scale;
            }
            for (int64_t s = t + 1; s < seq_len; ++s) {
                scores[s] = -std::numeric_limits<float>::infinity();
            }

            // Softmax over valid positions.
            float max_score = -std::numeric_limits<float>::infinity();
            for (int64_t s = 0; s < seq_len; ++s) {
                max_score = std::max(max_score, scores[s]);
            }
            float sum_exp = 0.0f;
            for (int64_t s = 0; s < seq_len; ++s) {
                weights[s] = std::exp(scores[s] - max_score);
                sum_exp += weights[s];
            }
            float inv_sum = 1.0f / sum_exp;
            for (int64_t s = 0; s < seq_len; ++s) {
                weights[s] *= inv_sum;
            }

            // Weighted sum of values.
            for (int64_t d = 0; d < head_dim; ++d) {
                out_head[d] = 0.0f;
            }
            for (int64_t s = 0; s <= t; ++s) {
                const float* v_head =
                    v_ptr + (s * num_kv_heads + kvh) * head_dim;
                float w = weights[s];
                for (int64_t d = 0; d < head_dim; ++d) {
                    out_head[d] += w * v_head[d];
                }
            }
            if (has_gate) {
                const float* gate_head = static_cast<const float*>(q_gate.data()) +
                    (t * num_q_heads + qh) * head_dim;
                for (int64_t d = 0; d < head_dim; ++d) {
                    out_head[d] *= 1.0f / (1.0f + std::exp(-gate_head[d]));
                }
            }
        }
    }

    Tensor attn_out_tensor(
        Shape{seq_len, num_q_heads * head_dim}, DType::FP32,
        input.device(), attn_out_buf);

    // 3. Output projection.
    Tensor output_input = mixed_bf16
        ? convert_dtype_host(attn_out_tensor, DType::BF16, backend.get())
        : attn_out_tensor;
    if (output_input.buffer() == nullptr) {
        return Tensor();
    }
    Tensor output = Linear::forward(output_input, wo, true, nullptr, stream);
    if (output.buffer() == nullptr) {
        return Tensor();
    }
    return mixed_bf16 ? convert_dtype_host(output, DType::FP32, backend.get())
                      : output;
}

} // namespace ops
} // namespace hybridai
