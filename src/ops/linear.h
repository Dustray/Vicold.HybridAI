#pragma once

#include "backends/interface/backend.h"
#include "core/status.h"
#include "core/tensor.h"

#include <cstdint>

namespace hybridai {
namespace ops {

// Linear (fully connected) operator: Y = X @ W^T + bias (optional).
// - input:  [M, K]
// - weight: [N, K] (row-major, not transposed by default)
// - output: [M, N]
// If transpose_weight is true, weight is interpreted as [K, N] and is
// transposed internally (used when weight storage layout is [K, N]).
// Supports adding an optional bias of shape [N].
class Linear {
public:
    Linear() = default;
    Linear(const Linear&) = default;
    Linear(Linear&&) = default;
    Linear& operator=(const Linear&) = default;
    Linear& operator=(Linear&&) = default;

    // Stateless forward. Returns an empty Tensor on error.
    static Tensor forward(const Tensor& input, const Tensor& weight,
                          bool transpose_weight = true,
                          const Tensor* bias = nullptr,
                          Stream* stream = nullptr);

    // Compute into a caller-owned output buffer. The buffer must be large
    // enough for the contiguous output tensor and must belong to the same
    // device as the input. This is intended for inference workspaces that
    // reuse temporary projection storage across calls.
    static Tensor forward_into(const Tensor& input, const Tensor& weight,
                               const std::shared_ptr<Buffer>& output_buffer,
                               bool transpose_weight = true,
                               const Tensor* bias = nullptr,
                               Stream* stream = nullptr);

    // Compute output shape from input and weight shapes.
    static Status compute_output_shape(const Shape& input_shape,
                                       const Shape& weight_shape,
                                       bool transpose_weight,
                                       Shape* output_shape);

    // Validate argument compatibility.
    static Status validate(const Tensor& input, const Tensor& weight,
                           bool transpose_weight, const Tensor* bias);
};

} // namespace ops
} // namespace hybridai
