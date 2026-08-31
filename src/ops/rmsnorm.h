#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

// RMSNorm(x) = x / sqrt(mean(x^2) + eps) * weight
// Input shape: [..., hidden_size]
// Weight shape: [hidden_size]
class RMSNorm {
public:
    static Tensor forward(const Tensor& input, const Tensor& weight,
                          float eps = 1e-6f, Stream* stream = nullptr,
                          bool add_unit_offset = false);

    // Validate shapes/dtype/device.
    static Status validate(const Tensor& input, const Tensor& weight,
                         float eps);
};

} // namespace ops
} // namespace hybridai
