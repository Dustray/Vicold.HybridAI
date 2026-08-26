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
#include <vector>

namespace hybridai {
namespace ops {

Status GatedGQAAttention::validate(const Tensor& input,
                                    const Tensor& wq, const Tensor& wk,
                                    const Tensor& wv, const Tensor& wo,
                                    int64_t num_q_heads, int64_t num_kv_heads,
                                    int64_t head_dim, int64_t rope_head_dim,
                                    float rope_base) {
    (void)rope_base;
    if (input.dtype() != DType::FP32) {
        return Status(StatusCode::UnsupportedDType,
                        "GatedGQAAttention CPU only supports FP32");
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
    if (wq.shape().dim(0) != q_out) {
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
                                 float rope_base, Stream* stream) {
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

    // 1. Project Q/K/V.
    Tensor q = Linear::forward(input, wq, true, nullptr, stream);
    Tensor k = Linear::forward(input, wk, true, nullptr, stream);
    Tensor v = Linear::forward(input, wv, true, nullptr, stream);
    if (q.data() == nullptr || k.data() == nullptr || v.data() == nullptr) {
        return Tensor();
    }

    // Reshape to [seq_len, num_heads, head_dim] and apply RoPE.
    Tensor q_3d(q.reshape(Shape{seq_len, num_q_heads, head_dim}));
    Tensor k_3d(k.reshape(Shape{seq_len, num_kv_heads, head_dim}));

    Tensor q_rot = RoPE::forward(q_3d, head_dim, rope_base, stream);
    Tensor k_rot = RoPE::forward(k_3d, head_dim, rope_base, stream);
    if (q_rot.data() == nullptr || k_rot.data() == nullptr) {
        return Tensor();
    }

    // 2. Compute attention scores and output.
    // For GQA each KV head is shared by q_heads_per_kv_head query heads.
    const int64_t q_heads_per_kv = num_q_heads / num_kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

    // Buffer for concatenated attention output [seq_len, num_q_heads, head_dim].
    auto backend = BackendRegistry::instance().create_backend(input.device());
    if (backend == nullptr) {
        return Tensor();
    }
    const size_t attn_out_bytes =
        static_cast<size_t>(seq_len * num_q_heads * head_dim) * sizeof(float);
    auto attn_out_buf = backend->create_buffer(attn_out_bytes,
                                               input.buffer()->memory_type());
    if (attn_out_buf == nullptr) {
        return Tensor();
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
        }
    }

    Tensor attn_out_tensor(
        Shape{seq_len, num_q_heads * head_dim}, DType::FP32,
        input.device(), attn_out_buf);

    // 3. Output projection.
    return Linear::forward(attn_out_tensor, wo, true, nullptr, stream);
}

} // namespace ops
} // namespace hybridai
