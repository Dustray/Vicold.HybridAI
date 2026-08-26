#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

// Rotary Position Embedding (RoPE) for 2-D input [seq_len, num_heads * head_dim]
// or [batch * seq_len, num_heads * head_dim]. Applies rotations per head
// independently. Supports in-place (output == input buffer) or out-of-place.
class RoPE {
public:
    // q/k: tensor of shape [..., num_heads * head_dim]
    // seq_len: sequence length used to compute theta frequencies
    // head_dim: per-head dimension; must be even
    // base: theta base, commonly 10000.0f or 1000000.0f for long-context
    //
    // Returns a tensor of the same shape with RoPE applied.
    static Tensor forward(const Tensor& input, int64_t head_dim,
                        float base = 10000.0f, Stream* stream = nullptr);

    static Status validate(const Tensor& input, int64_t head_dim, float base);
};

} // namespace ops
} // namespace hybridai
