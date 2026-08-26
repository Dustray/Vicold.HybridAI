#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

// Gated DeltaNet (linear attention) for Qwen3.8.
//
// Hyperparameters from Qwen3.8-27B-FP8:
//   hidden_size = 5120
//   num_qk_heads = 16, num_v_heads = 48, head_dim = 128
//
// This is a simplified recurrent reference implementation for Phase 5:
//   q, k, v projections
//   apply gating/decay per head
//   recurrent state update: S_t = diag(gate_t) * S_{t-1} + k_t^T v_t
//   o_t = q_t @ S_t
//   output projection
//
// The parallel/welcome form and full DeltaNet gate mechanisms are left as later
// optimizations. This operator is sufficient for functional end-to-end tests.
class GatedDeltaNet {
public:
    static Tensor forward(const Tensor& input,
                          const Tensor& wq, const Tensor& wk,
                          const Tensor& wv, const Tensor& wo,
                          int64_t num_qk_heads, int64_t num_v_heads,
                          int64_t head_dim,
                          Stream* stream = nullptr);

    static Status validate(const Tensor& input,
                           const Tensor& wq, const Tensor& wk,
                           const Tensor& wv, const Tensor& wo,
                           int64_t num_qk_heads, int64_t num_v_heads,
                           int64_t head_dim);
};

} // namespace ops
} // namespace hybridai
