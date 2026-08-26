#include "core/tensor.h"

#include "backends/backend_registry.h"

#include <algorithm>
#include <cstring>
#include <numeric>

namespace hybridai {

namespace {

// Generic element copy for contiguous -> contiguous between same dtype.
Status copy_buffer(Buffer* dst, const Buffer* src, size_t size) {
    if (dst == nullptr || src == nullptr) {
        return Status(StatusCode::InvalidArgument, "Null buffer in copy");
    }
    std::memcpy(dst->data(), src->data(), size);
    return Status::OK();
}

} // namespace

void Tensor::compute_strides() {
    strides_.resize(shape_.ndim());
    if (shape_.ndim() == 0) {
        is_contiguous_ = true;
        return;
    }
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.ndim()) - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_.dim(static_cast<size_t>(i));
    }
    is_contiguous_ = true;
}

void Tensor::check_contiguous() {
    if (shape_.ndim() == 0) {
        is_contiguous_ = true;
        return;
    }
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.ndim()) - 1; i >= 0; --i) {
        if (strides_[i] != stride) {
            is_contiguous_ = false;
            return;
        }
        stride *= shape_.dim(static_cast<size_t>(i));
    }
    is_contiguous_ = true;
}

Tensor Tensor::to(Device target) const {
    if (device_ == target) {
        return *this;
    }

    if (buffer_ == nullptr) {
        return Tensor(shape_, dtype_, target, nullptr);
    }

    auto backend = BackendRegistry::instance().create_backend(target);
    if (backend == nullptr) {
        return Tensor(); // invalid/unsupported target
    }

    MemoryType mem_type = target.is_cpu()
                              ? MemoryType::Host
                              : MemoryType::Device;
    auto dst_buffer = backend->create_buffer(nbytes(), mem_type);
    if (dst_buffer == nullptr) {
        return Tensor(); // allocation failed
    }

    Status status = backend->copy(dst_buffer.get(), buffer_.get(), nbytes());
    if (!status.ok()) {
        return Tensor();
    }

    return Tensor(shape_, dtype_, target, std::move(dst_buffer));
}

Tensor Tensor::contiguous() const {
    if (is_contiguous_) {
        return *this;
    }

    if (buffer_ == nullptr) {
        return *this;
    }

    auto backend = BackendRegistry::instance().create_backend(device_);
    if (backend == nullptr) {
        return Tensor();
    }

    auto dst_buffer = backend->create_buffer(nbytes(), buffer_->memory_type());
    if (dst_buffer == nullptr) {
        return Tensor();
    }

    const int64_t n = numel();
    const size_t elem_size = SizeOfDType(dtype_);

    // Generic strided -> contiguous copy. Optimized for common cases below.
    if (strides_.size() == 1) {
        // 1-D: just copy with stride
        const char* src = static_cast<const char*>(buffer_->data());
        char* dst = static_cast<char*>(dst_buffer->data());
        const int64_t stride = strides_[0] * static_cast<int64_t>(elem_size);
        for (int64_t i = 0; i < n; ++i) {
            std::memcpy(dst + i * elem_size, src + i * stride, elem_size);
        }
    } else if (strides_.size() == 2) {
        const char* src = static_cast<const char*>(buffer_->data());
        char* dst = static_cast<char*>(dst_buffer->data());
        const int64_t d0 = shape_.dim(0);
        const int64_t d1 = shape_.dim(1);
        const int64_t s0 = strides_[0];
        const int64_t s1 = strides_[1];
        for (int64_t i = 0; i < d0; ++i) {
            for (int64_t j = 0; j < d1; ++j) {
                const int64_t src_idx = i * s0 + j * s1;
                const int64_t dst_idx = i * d1 + j;
                std::memcpy(dst + dst_idx * elem_size,
                            src + src_idx * elem_size, elem_size);
            }
        }
    } else {
        // General N-D fallback using recursive index generation
        const char* src = static_cast<const char*>(buffer_->data());
        char* dst = static_cast<char*>(dst_buffer->data());
        const size_t ndim = strides_.size();
        std::vector<int64_t> indices(ndim, 0);
        for (int64_t flat = 0; flat < n; ++flat) {
            int64_t src_idx = 0;
            for (size_t d = 0; d < ndim; ++d) {
                src_idx += indices[d] * strides_[d];
            }
            std::memcpy(dst + flat * elem_size, src + src_idx * elem_size,
                        elem_size);
            // Increment indices (row-major)
            for (int64_t d = static_cast<int64_t>(ndim) - 1; d >= 0; --d) {
                ++indices[static_cast<size_t>(d)];
                if (indices[static_cast<size_t>(d)] < shape_.dim(d)) {
                    break;
                }
                indices[static_cast<size_t>(d)] = 0;
            }
        }
    }

    Tensor result(shape_, dtype_, device_, std::move(dst_buffer));
    return result;
}

} // namespace hybridai
