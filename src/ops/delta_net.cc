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
    auto backend = BackendRegistry::instance().create_backend(input.device());
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

} // namespace ops
} // namespace hybridai
