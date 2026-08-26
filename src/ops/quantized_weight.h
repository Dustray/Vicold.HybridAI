#pragma once

#include "core/status.h"
#include "core/tensor.h"

#include <cstdint>

namespace hybridai::ops {

enum class ScaleLayout : uint8_t {
    None,
    PerTensor,
    PerRow,
    PerColumn,
    Block2D,
};

struct QuantizationSpec {
    ScaleLayout scale_layout = ScaleLayout::None;
    int64_t block_rows = 1;
    int64_t block_columns = 1;
};

struct QuantizedWeight {
    Tensor values;
    Tensor scales;
    QuantizationSpec quantization;

    bool has_scales() const noexcept { return scales.buffer() != nullptr; }

    Status validate() const;
};

} // namespace hybridai::ops