#include "ops/quantized_weight.h"

namespace hybridai::ops {

Status QuantizedWeight::validate() const {
    if (values.buffer() == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "Weight values are not loaded");
    }
    if (quantization.scale_layout == ScaleLayout::None) {
        if (scales.buffer() != nullptr) {
            return Status(StatusCode::InvalidArgument,
                          "Scales are present but scale layout is None");
        }
        if (values.dtype() != DType::FP32 && values.dtype() != DType::FP16 &&
            values.dtype() != DType::BF16) {
            return Status(StatusCode::InvalidArgument,
                          "Dense weight must be FP32, FP16, or BF16");
        }
        return Status::OK();
    }
    if (values.dtype() != DType::FP8_E4M3) {
        return Status(StatusCode::InvalidArgument,
                      "Scaled weight values must be FP8_E4M3");
    }
    if (scales.buffer() == nullptr || scales.dtype() != DType::FP32) {
        return Status(StatusCode::InvalidArgument,
                      "QuantizedWeight scales must be FP32");
    }
    if (values.shape().ndim() != 2) {
        return Status(StatusCode::InvalidArgument,
                      "Scaled quantized weights must be matrices");
    }
    const int64_t rows = values.shape().dim(0);
    const int64_t columns = values.shape().dim(1);
    if (quantization.scale_layout == ScaleLayout::PerTensor &&
        scales.numel() != 1) {
        return Status(StatusCode::InvalidArgument,
                      "Per-tensor scale must contain one value");
    }
    if (quantization.scale_layout == ScaleLayout::PerRow &&
        scales.numel() != rows) {
        return Status(StatusCode::InvalidArgument,
                      "Per-row scale count does not match rows");
    }
    if (quantization.scale_layout == ScaleLayout::PerColumn &&
        scales.numel() != columns) {
        return Status(StatusCode::InvalidArgument,
                      "Per-column scale count does not match columns");
    }
    if (quantization.scale_layout == ScaleLayout::Block2D) {
        if (quantization.block_rows <= 0 || quantization.block_columns <= 0) {
            return Status(StatusCode::InvalidArgument,
                          "Block dimensions must be positive");
        }
        const int64_t block_rows =
            (rows + quantization.block_rows - 1) / quantization.block_rows;
        const int64_t block_columns =
            (columns + quantization.block_columns - 1) /
            quantization.block_columns;
        if (scales.shape() != Shape({block_rows, block_columns})) {
            return Status(StatusCode::InvalidArgument,
                          "2D scale shape does not match weight blocks");
        }
    }
    return Status::OK();
}

} // namespace hybridai::ops