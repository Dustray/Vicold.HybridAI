#pragma once

#include "core/status.h"
#include "core/tensor.h"

namespace hybridai {
namespace ops {

// Elementwise unary operators. Currently CPU-only FP32 reference
// implementations; GPU kernels can be registered via KernelRegistry.
class Elementwise {
public:
    // ReLU(x) = max(0, x)
    static Tensor relu(const Tensor& input, Stream* stream = nullptr);

    // SiLU(x) = x * sigmoid(x)
    static Tensor silu(const Tensor& input, Stream* stream = nullptr);

    // GELU(x) approximated with tanh (HuggingFace default).
    static Tensor gelu(const Tensor& input, Stream* stream = nullptr);

    // SwiGLU variant: Swish-beta(x) = x * sigmoid(beta * x)
    static Tensor swish(const Tensor& input, float beta = 1.0f,
                        Stream* stream = nullptr);

    enum class UnaryOp { ReLU, SiLU, GELU, Swish };

    static Tensor unary(const Tensor& input, UnaryOp op, float param,
                        Stream* stream);
};

} // namespace ops
} // namespace hybridai
