#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace hybridai {

class Shape {
public:
    Shape() = default;
    Shape(std::initializer_list<int64_t> dims) : dims_(dims) {}
    explicit Shape(std::vector<int64_t> dims) : dims_(std::move(dims)) {}

    size_t ndim() const noexcept { return dims_.size(); }
    int64_t dim(size_t idx) const { return dims_.at(idx); }
    const std::vector<int64_t>& dims() const noexcept { return dims_; }

    int64_t numel() const {
        if (dims_.empty()) return 0;
        return std::accumulate(dims_.begin(), dims_.end(), int64_t{1},
                               std::multiplies<int64_t>());
    }

    bool operator==(const Shape& other) const noexcept {
        return dims_ == other.dims_;
    }
    bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

private:
    std::vector<int64_t> dims_;
};

} // namespace hybridai
