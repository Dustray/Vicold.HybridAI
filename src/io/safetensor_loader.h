#pragma once

#include "core/device.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/status.h"
#include "core/tensor.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hybridai::io {

struct SafeTensorInfo {
    DType dtype = DType::FP32;
    Shape shape;
    uint64_t data_begin = 0;
    uint64_t data_end = 0;
};

class SafeTensorLoader {
public:
    Status open(const std::string& path);

    bool is_open() const noexcept { return !path_.empty(); }
    const std::string& path() const noexcept { return path_; }

    std::vector<std::string> tensor_names() const;
    const SafeTensorInfo* tensor_info(const std::string& name) const;

    Status load(const std::string& name, Device device, Tensor* output,
                MemoryType memory_type = MemoryType::Host) const;

private:
    std::string path_;
    uint64_t data_section_offset_ = 0;
    std::unordered_map<std::string, SafeTensorInfo> tensors_;
};

} // namespace hybridai::io
