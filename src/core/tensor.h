#pragma once

#include "backends/interface/backend.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"

#include <memory>
#include <utility>
#include <vector>

namespace hybridai {

class Tensor {
public:
    Tensor() = default;

    Tensor(const Shape& shape, DType dtype, Device device,
           std::shared_ptr<Buffer> buffer)
        : shape_(shape), dtype_(dtype), device_(device),
          buffer_(std::move(buffer)) {
        compute_strides();
    }

    Tensor(const Shape& shape, DType dtype, Device device,
           std::shared_ptr<Buffer> buffer, const std::vector<int64_t>& strides)
        : shape_(shape), dtype_(dtype), device_(device),
          buffer_(std::move(buffer)), strides_(strides) {
        check_contiguous();
    }

    int64_t numel() const { return shape_.numel(); }
    size_t nbytes() const {
        return static_cast<size_t>(numel()) * SizeOfDType(dtype_);
    }

    const Shape& shape() const noexcept { return shape_; }
    DType dtype() const noexcept { return dtype_; }
    Device device() const noexcept { return device_; }
    const std::vector<int64_t>& strides() const noexcept { return strides_; }

    bool is_contiguous() const noexcept { return is_contiguous_; }

    void* data() { return buffer_ ? buffer_->data() : nullptr; }
    const void* data() const { return buffer_ ? buffer_->data() : nullptr; }

    std::shared_ptr<Buffer> buffer() const noexcept { return buffer_; }

    void swap(Tensor& other) noexcept {
        std::swap(shape_, other.shape_);
        std::swap(dtype_, other.dtype_);
        std::swap(device_, other.device_);
        buffer_.swap(other.buffer_);
        strides_.swap(other.strides_);
        std::swap(is_contiguous_, other.is_contiguous_);
    }

    // Move tensor to another device (uses backend copy)
    Tensor to(Device target) const;

    // Make contiguous copy if strides are not standard
    Tensor contiguous() const;

    // Return a new tensor view with the given shape and the same buffer.
    // The total number of elements must match. Strides are recomputed as
    // contiguous; this is a convenience for operators that need to reshape
    // between [seq_len, num_heads * head_dim] and [seq_len, num_heads, head_dim].
    Tensor reshape(const Shape& new_shape) const;

    bool operator==(const Tensor& other) const noexcept {
        return shape_ == other.shape_ && dtype_ == other.dtype_ &&
               device_ == other.device_ && buffer_ == other.buffer_;
    }

private:
    void compute_strides();
    void check_contiguous();

    Shape shape_;
    DType dtype_;
    Device device_;
    std::shared_ptr<Buffer> buffer_;
    std::vector<int64_t> strides_;
    bool is_contiguous_ = true;
};

} // namespace hybridai
