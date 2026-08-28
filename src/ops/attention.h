#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

// Gated GQA Attention for Qwen3.8 (standard attention layer).
//
// Hyperparameters from Qwen3.8-27B-FP8:
//   hidden_size = 5120
//   num_q_heads = 24, num_kv_heads = 4, head_dim = 256
//   rope_head_dim = 64 (only first 64 dims of each head get RoPE)
//
// This operator performs:
//   q = input @ Wq.T      -> [seq_len, num_q_heads * head_dim]
//   k = input @ Wk.T      -> [seq_len, num_kv_heads * head_dim]
//   v = input @ Wv.T      -> [seq_len, num_kv_heads * head_dim]
//   apply RoPE to q and k over rope_head_dim
//   scores = q @ k.T / sqrt(head_dim)   (GQA: expand KV heads)
//   attn = softmax(scores) @ v
//   output = attn @ Wo.T
//
// For Phase 4/5 this is a reference CPU FP32 implementation. The interface is
// kept backend-agnostic; GPU kernels can be registered via KernelRegistry.
class GatedGQAAttention {
public:
    // Stateless forward. All weight tensors must be on the same device as input.
    static Tensor forward(const Tensor& input,
                          const Tensor& wq, const Tensor& wk,
                          const Tensor& wv, const Tensor& wo,
                          int64_t num_q_heads, int64_t num_kv_heads,
                          int64_t head_dim, int64_t rope_head_dim,
                          float rope_base = 10000.0f,
                          Stream* stream = nullptr,
                          const Tensor& q_norm = Tensor(),
                          const Tensor& k_norm = Tensor(),
                          float rms_norm_eps = 1.0e-6f);

    static Status validate(const Tensor& input,
                           const Tensor& wq, const Tensor& wk,
                           const Tensor& wv, const Tensor& wo,
                           int64_t num_q_heads, int64_t num_kv_heads,
                           int64_t head_dim, int64_t rope_head_dim,
                           float rope_base);
};

} // namespace ops
} // namespace hybridai
