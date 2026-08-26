#include "ops/rmsnorm.h"

#include "backends/backend_registry.h"
#include "ops/registry.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hybridai {
namespace ops {

Status RMSNorm::validate(const Tensor& input, const Tensor& weight, float eps) {
    (void)eps;
    if (input.dtype() != weight.dtype()) {
        return Status(StatusCode::InvalidArgument,
                        "RMSNorm input/weight dtype mismatch");
    }
    if (input.device() != weight.device()) {
        return Status(StatusCode::InvalidArgument,
                        "RMSNorm input/weight must be on the same device");
    }
    if (input.shape().ndim() == 0) {
        return Status(StatusCode::InvalidArgument,
                        "RMSNorm input must have at least one dimension");
    }
    const int64_t hidden_size = input.shape().dim(input.shape().ndim() - 1);
    if (weight.shape().ndim() != 1 || weight.shape().dim(0) != hidden_size) {
        return Status(StatusCode::InvalidArgument,
                        "RMSNorm weight must be 1-D [hidden_size]");
    }
    return Status::OK();
}

Tensor RMSNorm::forward(const Tensor& input, const Tensor& weight,
                        float eps, Stream* stream) {
    (void)stream;
    Status status = validate(input, weight, eps);
    if (!status.ok()) {
        return Tensor();
    }

    auto backend = BackendRegistry::instance().create_backend(input.device());
    if (backend == nullptr) {
        return Tensor();
    }

    auto output_buffer = backend->create_buffer(
        input.nbytes(), input.buffer()->memory_type());
    if (output_buffer == nullptr) {
        return Tensor();
    }

    const size_t ndim = input.shape().ndim();
    const int64_t hidden_size = input.shape().dim(ndim - 1);
    int64_t num_rows = 1;
    for (size_t i = 0; i + 1 < ndim; ++i) {
        num_rows *= input.shape().dim(i);
    }

    const float* src = static_cast<const float*>(input.data());
    const float* w = static_cast<const float*>(weight.data());
    float* dst = static_cast<float*>(output_buffer->data());

    for (int64_t r = 0; r < num_rows; ++r) {
        const float* row_in = src + r * hidden_size;
        float* row_out = dst + r * hidden_size;

        float sum_sq = 0.0f;
        for (int64_t i = 0; i < hidden_size; ++i) {
            sum_sq += row_in[i] * row_in[i];
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(hidden_size) + eps);
        float inv_rms = 1.0f / rms;

        for (int64_t i = 0; i < hidden_size; ++i) {
            row_out[i] = row_in[i] * inv_rms * w[i];
        }
    }

    return Tensor(input.shape(), input.dtype(), input.device(),
                  std::move(output_buffer));
}

} // namespace ops
} // namespace hybridai
