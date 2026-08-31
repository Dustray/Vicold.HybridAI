#include "ops/linear.h"

#include "backends/backend_registry.h"
#include "ops/registry.h"

#include <cmath>
#include <cstdint>
#include <limits>

namespace hybridai {
namespace ops {

Status Linear::compute_output_shape(const Shape& input_shape,
                                  const Shape& weight_shape,
                                  bool transpose_weight,
                                  Shape* output_shape) {
    if (input_shape.ndim() != 2 || weight_shape.ndim() != 2) {
        return Status(StatusCode::InvalidArgument,
                        "Linear supports only 2-D tensors");
    }

    const int64_t m = input_shape.dim(0);
    const int64_t k_input = input_shape.dim(1);
    const int64_t k_weight = transpose_weight ? weight_shape.dim(1)
                                             : weight_shape.dim(0);
    const int64_t n = transpose_weight ? weight_shape.dim(0)
                                         : weight_shape.dim(1);

    if (k_input != k_weight) {
        return Status(StatusCode::InvalidArgument,
                        "Linear input/weight K dimension mismatch");
    }

    *output_shape = Shape{m, n};
    return Status::OK();
}

Status Linear::validate(const Tensor& input, const Tensor& weight,
                        bool transpose_weight, const Tensor* bias) {
    Shape unused;
    Status status = compute_output_shape(input.shape(), weight.shape(),
                                           transpose_weight, &unused);
    if (!status.ok()) return status;

    const bool supported_mixed_types =
        input.dtype() == DType::FP32 && weight.dtype() == DType::BF16;
    if (input.dtype() != weight.dtype() && !supported_mixed_types) {
        return Status(StatusCode::InvalidArgument,
                        "Linear input/weight dtype mismatch (only FP32 x BF16 is supported)");
    }

    if (input.device() != weight.device()) {
        return Status(StatusCode::InvalidArgument,
                        "Linear input/weight must be on the same device");
    }

    if (bias != nullptr) {
        if (bias->dtype() != input.dtype()) {
            return Status(StatusCode::InvalidArgument,
                            "Linear bias dtype mismatch");
        }
        if (bias->device() != input.device()) {
            return Status(StatusCode::InvalidArgument,
                            "Linear bias must be on the same device as input");
        }
        if (bias->shape().ndim() != 1) {
            return Status(StatusCode::InvalidArgument,
                            "Linear bias must be 1-D");
        }
        const int64_t n = transpose_weight ? weight.shape().dim(0)
                                            : weight.shape().dim(1);
        if (bias->shape().dim(0) != n) {
            return Status(StatusCode::InvalidArgument,
                            "Linear bias size mismatch");
        }
    }

    return Status::OK();
}

Tensor Linear::forward(const Tensor& input, const Tensor& weight,
                       bool transpose_weight, const Tensor* bias,
                       Stream* stream) {
    Status status = validate(input, weight, transpose_weight, bias);
    if (!status.ok()) {
        return Tensor();
    }

    const int64_t m = input.shape().dim(0);
    const int64_t k = input.shape().dim(1);
    const int64_t n = transpose_weight ? weight.shape().dim(0)
                                         : weight.shape().dim(1);

    // Use the kernel registry to allow custom tile kernels to override BLAS.
    // If no kernel is registered, fall back to backend gemm().
    KernelKey key{"linear", input.device().type(), input.dtype()};
    KernelFn kernel = KernelRegistry::instance().find_kernel(key);
    if (kernel != nullptr) {
        // TODO: define tensor-based kernel signature for Linear.
        // For now fall through to backend gemm.
    }

    auto backend = BackendRegistry::instance().get_backend(input.device());
    if (backend == nullptr) {
        return Tensor();
    }

    Shape output_shape;
    status = compute_output_shape(input.shape(), weight.shape(),
                                    transpose_weight, &output_shape);
    if (!status.ok()) {
        return Tensor();
    }

    const DType output_dtype = input.dtype();
    auto output_buffer = backend->create_buffer(
        output_shape.numel() * SizeOfDType(output_dtype),
        input.buffer()->memory_type());
    if (output_buffer == nullptr) {
        return Tensor();
    }

    // GEMM: C = A * B, where A = input [M, K], B = weight^T [K, N]
    //       => C [M, N].
    // Weight storage is [N, K] when transpose_weight is true, so B is weight^T.
    // rocBLAS/cuBLAS interpret the matrices as column-major by default.
    // We pass trans_a=false, trans_b=true and let the backend handle layout.
    status = backend->gemm(output_buffer.get(), input.buffer().get(),
                           weight.buffer().get(),
                           output_dtype, input.dtype(), weight.dtype(),
                           output_dtype == DType::BF16 ? DType::FP32
                                                        : output_dtype,
                           /*trans_a=*/false, /*trans_b=*/transpose_weight,
                           m, n, k, 1.0f, 0.0f, stream);
    if (!status.ok()) {
        return Tensor();
    }

    if (bias != nullptr) {
        if (input.device().is_gpu()) {
            status = backend->add_row_bias(output_buffer.get(),
                                           bias->buffer().get(), output_dtype,
                                           m, n, stream);
            if (!status.ok()) return Tensor();
        } else {
            if (output_dtype != DType::FP32) return Tensor();
            float* output = static_cast<float*>(output_buffer->data());
            const float* bias_data = static_cast<const float*>(bias->data());
            for (int64_t row = 0; row < m; ++row) {
                for (int64_t column = 0; column < n; ++column) {
                    output[row * n + column] += bias_data[column];
                }
            }
        }
    }

    return Tensor(output_shape, output_dtype, input.device(),
                  std::move(output_buffer));
}

} // namespace ops
} // namespace hybridai
