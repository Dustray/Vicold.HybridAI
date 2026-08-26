#pragma once

#include "core/dtype.h"
#include "core/status.h"
#include "core/tensor.h"

#include <cstddef>
#include <cstdint>

namespace hybridai::ops {

class FP8Dequantizer {
public:
    // Dequantize an FP8 E4M3 tensor to FP32. Scales are optional. If present,
    // one scale is applied to each contiguous block of `block_size` values.
    static Status dequantize(const Tensor& input, Tensor* output,
                             const Tensor* scales = nullptr,
                             int64_t block_size = 128);

    static Status dequantize_2d(const Tensor& input, const Tensor& scales,
                                Tensor* output, int64_t block_rows = 128,
                                int64_t block_columns = 128);

    static float decode_e4m3(uint8_t value) noexcept;
};

} // namespace hybridai::ops
