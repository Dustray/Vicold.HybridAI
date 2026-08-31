#include "ops/elementwise.h"

#include "backends/backend_registry.h"
#include "ops/registry.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hybridai {
namespace ops {

namespace {

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

using hybridai::ops::Elementwise;

inline float apply_op(float x, Elementwise::UnaryOp op, float param) {
    switch (op) {
        case Elementwise::UnaryOp::ReLU:
            return std::max(0.0f, x);
        case Elementwise::UnaryOp::SiLU:
            return x * sigmoid(x);
        case Elementwise::UnaryOp::GELU:
            // Erf approximation used by many frameworks:
            // 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            return 0.5f * x *
                   (1.0f +
                    std::tanh(
                        0.7978845608f *
                        (x + 0.0447149983f * x * x * x)));
        case Elementwise::UnaryOp::Swish:
            return x * sigmoid(param * x);
    }
    return x;
}

Status validate_unary(const Tensor& input) {
    if (input.dtype() != DType::FP32 &&
        !(input.device().is_gpu() && input.dtype() == DType::BF16)) {
        return Status(StatusCode::UnsupportedDType,
                        "Elementwise supports CPU FP32 and GPU FP32/BF16");
    }
    if (input.buffer() == nullptr || input.data() == nullptr) {
        return Status(StatusCode::InvalidArgument,
                        "Elementwise input has no data");
    }
    return Status::OK();
}

} // namespace

Tensor Elementwise::unary(const Tensor& input, UnaryOp op, float param,
                          Stream* stream) {
    (void)stream;
    Status status = validate_unary(input);
    if (!status.ok()) {
        return Tensor();
    }

    auto backend = BackendRegistry::instance().get_backend(input.device());
    if (backend == nullptr) {
        return Tensor();
    }

    auto output_buffer = backend->create_buffer(
        input.nbytes(), input.buffer()->memory_type());
    if (output_buffer == nullptr) {
        return Tensor();
    }

    const int64_t n = input.numel();
    if (input.device().is_gpu()) {
        status = backend->unary(output_buffer.get(), input.buffer().get(),
                                input.dtype(), n, static_cast<int>(op), param,
                                stream);
        if (!status.ok()) return Tensor();
        return Tensor(input.shape(), input.dtype(), input.device(),
                      std::move(output_buffer));
    }

    const float* src = static_cast<const float*>(input.data());
    float* dst = static_cast<float*>(output_buffer->data());

    for (int64_t i = 0; i < n; ++i) {
        dst[i] = apply_op(src[i], op, param);
    }

    return Tensor(input.shape(), input.dtype(), input.device(),
                  std::move(output_buffer));
}

Tensor Elementwise::relu(const Tensor& input, Stream* stream) {
    return unary(input, UnaryOp::ReLU, 0.0f, stream);
}

Tensor Elementwise::silu(const Tensor& input, Stream* stream) {
    return unary(input, UnaryOp::SiLU, 0.0f, stream);
}

Tensor Elementwise::gelu(const Tensor& input, Stream* stream) {
    return unary(input, UnaryOp::GELU, 0.0f, stream);
}

Tensor Elementwise::swish(const Tensor& input, float beta, Stream* stream) {
    return unary(input, UnaryOp::Swish, beta, stream);
}

Tensor Elementwise::silu_mul(const Tensor& gate, const Tensor& up,
                             Stream* stream) {
    if (gate.shape() != up.shape() || gate.dtype() != up.dtype() ||
        gate.device() != up.device() || gate.buffer() == nullptr ||
        up.buffer() == nullptr) {
        return Tensor();
    }
    auto backend = BackendRegistry::instance().get_backend(gate.device());
    if (backend == nullptr) return Tensor();
    auto output_buffer = backend->create_buffer(
        gate.nbytes(), gate.buffer()->memory_type());
    if (output_buffer == nullptr) return Tensor();

    if (gate.device().is_gpu()) {
        Status status = backend->silu_mul(
            output_buffer.get(), gate.buffer().get(), up.buffer().get(),
            gate.dtype(), gate.numel(), stream);
        if (!status.ok()) return Tensor();
        return Tensor(gate.shape(), gate.dtype(), gate.device(),
                      std::move(output_buffer));
    }
    if (gate.dtype() != DType::FP32) return Tensor();
    const float* gate_data = static_cast<const float*>(gate.data());
    const float* up_data = static_cast<const float*>(up.data());
    float* output = static_cast<float*>(output_buffer->data());
    for (int64_t i = 0; i < gate.numel(); ++i) {
        output[i] = apply_op(gate_data[i], UnaryOp::SiLU, 0.0f) * up_data[i];
    }
    return Tensor(gate.shape(), gate.dtype(), gate.device(),
                  std::move(output_buffer));
}

} // namespace ops
} // namespace hybridai
