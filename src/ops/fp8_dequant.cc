#include "ops/fp8_dequant.h"

#include "backends/backend_registry.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace hybridai::ops {

float FP8Dequantizer::decode_e4m3(uint8_t value) noexcept {
    const bool negative = (value & 0x80u) != 0;
    const int exponent = static_cast<int>((value >> 3u) & 0x0fu);
    const int mantissa = static_cast<int>(value & 0x07u);

    // E4M3FN: exponent 0 is subnormal/zero; exponents 1..14 are normal.
    float result;
    if (exponent == 0) {
        result = std::ldexp(static_cast<float>(mantissa), -9);
    } else {
        result = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                            exponent - 7);
    }
    return negative ? -result : result;
}

Status FP8Dequantizer::dequantize(const Tensor& input, Tensor* output,
                                  const Tensor* scales, int64_t block_size) {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "FP8Dequantizer output must not be null");
    }
    if (input.dtype() != DType::FP8_E4M3) {
        return Status(StatusCode::UnsupportedDType,
                      "FP8Dequantizer expects FP8_E4M3 input");
    }
    if (input.data() == nullptr || input.buffer() == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "FP8Dequantizer input has no data");
    }
    if (block_size <= 0) {
        return Status(StatusCode::InvalidArgument,
                      "FP8Dequantizer block_size must be positive");
    }
    if (scales != nullptr && (scales->data() == nullptr ||
                              scales->dtype() != DType::FP32)) {
        return Status(StatusCode::InvalidArgument,
                      "FP8Dequantizer scales must be an FP32 tensor");
    }

    auto backend = BackendRegistry::instance().create_backend(input.device());
    if (backend == nullptr) {
        return Status(StatusCode::InvalidDevice,
                      "No backend available for FP8 dequantization");
    }
    auto buffer = backend->create_buffer(input.numel() * sizeof(float),
                                         input.buffer()->memory_type());
    if (buffer == nullptr) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate FP8 dequantization output");
    }

    const auto* encoded = static_cast<const uint8_t*>(input.data());
    const float* scale_data = scales == nullptr
                                  ? nullptr
                                  : static_cast<const float*>(scales->data());
    float* decoded = static_cast<float*>(buffer->data());
    const int64_t scale_count = scales == nullptr ? 0 : scales->numel();

    for (int64_t i = 0; i < input.numel(); ++i) {
        float scale = 1.0f;
        if (scale_data != nullptr) {
            const int64_t scale_index = i / block_size;
            if (scale_index >= scale_count) {
                return Status(StatusCode::InvalidArgument,
                              "FP8Dequantizer scales do not cover input");
            }
            scale = scale_data[scale_index];
        }
        decoded[i] = decode_e4m3(encoded[i]) * scale;
    }

    *output = Tensor(input.shape(), DType::FP32, input.device(),
                     std::move(buffer));
    return Status::OK();
}

Status FP8Dequantizer::dequantize_2d(const Tensor& input,
                                     const Tensor& scales, Tensor* output,
                                     int64_t block_rows,
                                     int64_t block_columns) {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "FP8Dequantizer output must not be null");
    }
    if (input.dtype() != DType::FP8_E4M3 || scales.dtype() != DType::FP32) {
        return Status(StatusCode::UnsupportedDType,
                      "2D FP8 dequantization expects FP8 and FP32 scales");
    }
    if (input.shape().ndim() != 2 || scales.shape().ndim() != 2 ||
        input.data() == nullptr || scales.data() == nullptr ||
        block_rows <= 0 || block_columns <= 0) {
        return Status(StatusCode::InvalidArgument,
                      "Invalid 2D FP8 dequantization arguments");
    }
    const int64_t rows = input.shape().dim(0);
    const int64_t columns = input.shape().dim(1);
    const int64_t scale_rows = (rows + block_rows - 1) / block_rows;
    const int64_t scale_columns = (columns + block_columns - 1) /
                                  block_columns;
    if (scales.shape() != Shape({scale_rows, scale_columns})) {
        return Status(StatusCode::InvalidArgument,
                      "2D FP8 scale shape does not match input blocks");
    }

    auto backend = BackendRegistry::instance().create_backend(input.device());
    if (backend == nullptr) {
        return Status(StatusCode::InvalidDevice,
                      "No backend available for FP8 dequantization");
    }
    auto buffer = backend->create_buffer(input.numel() * sizeof(float),
                                         input.buffer()->memory_type());
    if (buffer == nullptr) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate FP8 dequantization output");
    }

    const auto* encoded = static_cast<const uint8_t*>(input.data());
    const auto* scale_data = static_cast<const float*>(scales.data());
    auto* decoded = static_cast<float*>(buffer->data());
    for (int64_t row = 0; row < rows; ++row) {
        for (int64_t column = 0; column < columns; ++column) {
            const int64_t scale_row = row / block_rows;
            const int64_t scale_column = column / block_columns;
            const int64_t scale_index = scale_row * scale_columns +
                                        scale_column;
            const int64_t index = row * columns + column;
            decoded[index] = decode_e4m3(encoded[index]) *
                             scale_data[scale_index];
        }
    }
    *output = Tensor(input.shape(), DType::FP32, input.device(),
                     std::move(buffer));
    return Status::OK();
}

} // namespace hybridai::ops
