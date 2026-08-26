#include "core/tensor.h"

#include <numeric>

namespace hybridai {

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

Tensor Tensor::to(Device target) const {
    // TODO: implement cross-device copy using Backend registry
    (void)target;
    return *this;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous_) {
        return *this;
    }
    // TODO: implement non-contiguous copy
    return *this;
}

} // namespace hybridai
