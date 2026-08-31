#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

struct DeltaNetCache {
    Tensor conv_state;
    Tensor recurrent_state;
    int64_t length = 0;

    bool initialized() const noexcept {
        return conv_state.buffer() != nullptr &&
               recurrent_state.buffer() != nullptr;
    }
    void reset() noexcept { length = 0; }
};

struct DeltaNetQKV {
    Tensor query;
    Tensor key;
    Tensor value;

    bool valid() const noexcept {
        return query.buffer() != nullptr && key.buffer() != nullptr &&
               value.buffer() != nullptr;
    }
};

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

        // Stateful GPU primitives used by Qwen DeltaNet prefill and decode.
        // causal_conv accepts the rearranged [q, k, v] stream and preserves the
        // last kernel_size - 1 raw inputs in cache->conv_state.
        static Tensor causal_conv(const Tensor& input, const Tensor& weight,
                                                            int64_t kernel_size, DeltaNetCache* cache,
                                                            Stream* stream = nullptr);

                                    static DeltaNetQKV grouped_causal_conv(
                                        const Tensor& grouped_qkv, const Tensor& weight,
                                        int64_t num_qk_heads, int64_t num_value_heads,
                                        int64_t key_head_dim, int64_t value_head_dim,
                                        int64_t kernel_size, DeltaNetCache* cache,
                                        Stream* stream = nullptr);

        // q/k/v/a/b/z are projected tensors for only the new tokens. b contains
        // beta logits. recurrent_state is always FP32 even when tensors are BF16.
        static Tensor recurrent(const Tensor& query, const Tensor& key,
                                                        const Tensor& value, const Tensor& a,
                                                        const Tensor& b, const Tensor& z,
                                                        const Tensor& a_log, const Tensor& dt_bias,
                                                        const Tensor& norm_weight,
                                                        int64_t num_qk_heads, int64_t num_value_heads,
                                                        int64_t key_head_dim, int64_t value_head_dim,
                                                        float eps, DeltaNetCache* cache,
                                                        Stream* stream = nullptr);

    static Status validate(const Tensor& input,
                           const Tensor& wq, const Tensor& wk,
                           const Tensor& wv, const Tensor& wo,
                           int64_t num_qk_heads, int64_t num_v_heads,
                           int64_t head_dim);
};

} // namespace ops
} // namespace hybridai
