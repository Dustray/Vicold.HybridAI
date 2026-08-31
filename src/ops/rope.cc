#include "ops/rope.h"

#include "backends/backend_registry.h"
#include "ops/registry.h"

#include <cmath>
#include <cstring>

namespace hybridai {
namespace ops {

Status RoPE::validate(const Tensor& input, int64_t head_dim, float base) {
    (void)base;
    if (head_dim <= 0 || (head_dim % 2) != 0) {
        return Status(StatusCode::InvalidArgument,
                        "RoPE head_dim must be positive and even");
    }
    if (input.shape().ndim() < 2) {
        return Status(StatusCode::InvalidArgument,
                        "RoPE input must have at least 2 dimensions");
    }
    const int64_t last_dim = input.shape().dim(input.shape().ndim() - 1);
    if (last_dim != head_dim) {
        return Status(StatusCode::InvalidArgument,
                        "RoPE input last dimension must equal head_dim");
    }
    const int64_t num_heads_dim = input.shape().dim(input.shape().ndim() - 2);
    if (num_heads_dim <= 0) {
        return Status(StatusCode::InvalidArgument,
                        "RoPE num_heads dimension must be positive");
    }
    return Status::OK();
}

Tensor RoPE::forward(const Tensor& input, int64_t head_dim,
                   float base, Stream* stream) {
    (void)stream;
    Status status = validate(input, head_dim, base);
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
    const int64_t num_heads = input.shape().dim(ndim - 2);
    const int64_t seq_len_actual =
        (ndim >= 3) ? input.shape().dim(ndim - 3) : 1;

    int64_t outer = 1;
    for (size_t i = 0; i + 3 < ndim; ++i) {
        outer *= input.shape().dim(i);
    }

    // Precompute inverse frequencies for the head_dim pairs.
    std::vector<float> inv_freq(head_dim / 2);
    for (int64_t j = 0; j < head_dim / 2; ++j) {
        inv_freq[j] = 1.0f / std::pow(base,
                                      2.0f * static_cast<float>(j) /
                                          static_cast<float>(head_dim));
    }

    const float* src = static_cast<const float*>(input.data());
    float* dst = static_cast<float*>(output_buffer->data());

    // Input layout: [outer..., seq_len, num_heads, head_dim].
    const int64_t stride_head = head_dim;
    const int64_t stride_seq = num_heads * head_dim;
    const int64_t stride_outer = seq_len_actual * stride_seq;

    for (int64_t b = 0; b < outer; ++b) {
        for (int64_t s = 0; s < seq_len_actual; ++s) {
            for (int64_t h = 0; h < num_heads; ++h) {
                const float* head_in =
                    src + b * stride_outer + s * stride_seq +
                    h * stride_head;
                float* head_out =
                    dst + b * stride_outer + s * stride_seq +
                    h * stride_head;

                for (int64_t j = 0; j < head_dim / 2; ++j) {
                    float theta =
                        static_cast<float>(s) * inv_freq[j];
                    float cos_t = std::cos(theta);
                    float sin_t = std::sin(theta);

                    float x0 = head_in[2 * j];
                    float x1 = head_in[2 * j + 1];

                    head_out[2 * j] = x0 * cos_t - x1 * sin_t;
                    head_out[2 * j + 1] = x0 * sin_t + x1 * cos_t;
                }
            }
        }
    }

    return Tensor(input.shape(), input.dtype(), input.device(),
                  std::move(output_buffer));
}

Tensor RoPE::partial(const Tensor& input, int64_t rope_head_dim,
                     int64_t position_offset, float base, Stream* stream) {
    if (input.buffer() == nullptr || input.shape().ndim() != 3 ||
        (input.dtype() != DType::FP32 && input.dtype() != DType::BF16) ||
        rope_head_dim <= 0 || (rope_head_dim % 2) != 0 ||
        rope_head_dim > input.shape().dim(2) || position_offset < 0 ||
        base <= 0.0f) {
        return Tensor();
    }
    auto backend = BackendRegistry::instance().get_backend(input.device());
    if (backend == nullptr) return Tensor();
    auto output_buffer = backend->create_buffer(input.nbytes(),
                                                 input.buffer()->memory_type());
    if (output_buffer == nullptr) return Tensor();
    const int64_t seq_len = input.shape().dim(0);
    const int64_t num_heads = input.shape().dim(1);
    const int64_t head_dim = input.shape().dim(2);
    if (input.device().type() != DeviceType::CPU) {
        Status status = backend->partial_rope(
            output_buffer.get(), input.buffer().get(), input.dtype(), seq_len,
            num_heads, head_dim, rope_head_dim, position_offset, base, stream);
        if (!status.ok()) return Tensor();
        return Tensor(input.shape(), input.dtype(), input.device(),
                      std::move(output_buffer));
    }
    if (input.dtype() != DType::FP32) return Tensor();
    const float* src = static_cast<const float*>(input.data());
    float* dst = static_cast<float*>(output_buffer->data());
    const int64_t half = rope_head_dim / 2;
    for (int64_t sequence = 0; sequence < seq_len; ++sequence) {
        for (int64_t head = 0; head < num_heads; ++head) {
            const float* source = src +
                (sequence * num_heads + head) * head_dim;
            float* target = dst + (sequence * num_heads + head) * head_dim;
            std::memcpy(target, source,
                        static_cast<size_t>(head_dim) * sizeof(float));
            for (int64_t dimension = 0; dimension < half; ++dimension) {
                const float theta =
                    static_cast<float>(position_offset + sequence) /
                    std::pow(base,
                             2.0f * static_cast<float>(dimension) /
                                 static_cast<float>(rope_head_dim));
                const float cosine = std::cos(theta);
                const float sine = std::sin(theta);
                const float x = source[dimension];
                const float y = source[dimension + half];
                target[dimension] = x * cosine - y * sine;
                target[dimension + half] = y * cosine + x * sine;
            }
        }
    }
    return Tensor(input.shape(), input.dtype(), input.device(),
                  std::move(output_buffer));
}

} // namespace ops
} // namespace hybridai
