#include "ops/softmax.h"

#include "backends/backend_registry.h"
#include "ops/registry.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace hybridai {
namespace ops {

Status Softmax::validate(const Tensor& input) {
    if (input.dtype() != DType::FP32) {
        return Status(StatusCode::UnsupportedDType,
                        "Softmax CPU only supports FP32");
    }
    if (input.shape().ndim() == 0) {
        return Status(StatusCode::InvalidArgument,
                        "Softmax input must have at least one dimension");
    }
    if (input.buffer() == nullptr || input.data() == nullptr) {
        return Status(StatusCode::InvalidArgument,
                        "Softmax input has no data");
    }
    return Status::OK();
}

Tensor Softmax::forward(const Tensor& input, Stream* stream) {
    (void)stream;
    Status status = validate(input);
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

    const size_t ndim = input.shape().ndim();
    const int64_t vocab_size = input.shape().dim(ndim - 1);
    int64_t num_rows = 1;
    for (size_t i = 0; i + 1 < ndim; ++i) {
        num_rows *= input.shape().dim(i);
    }

    const float* src = static_cast<const float*>(input.data());
    float* dst = static_cast<float*>(output_buffer->data());

    for (int64_t r = 0; r < num_rows; ++r) {
        const float* row_in = src + r * vocab_size;
        float* row_out = dst + r * vocab_size;

        float max_val = -std::numeric_limits<float>::infinity();
        for (int64_t i = 0; i < vocab_size; ++i) {
            max_val = std::max(max_val, row_in[i]);
        }

        float sum_exp = 0.0f;
        for (int64_t i = 0; i < vocab_size; ++i) {
            float e = std::exp(row_in[i] - max_val);
            row_out[i] = e;
            sum_exp += e;
        }

        float inv_sum = 1.0f / sum_exp;
        for (int64_t i = 0; i < vocab_size; ++i) {
            row_out[i] *= inv_sum;
        }
    }

    return Tensor(input.shape(), input.dtype(), input.device(),
                  std::move(output_buffer));
}

} // namespace ops
} // namespace hybridai
