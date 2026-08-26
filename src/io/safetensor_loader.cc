#include "io/safetensor_loader.h"

#include "backends/backend_registry.h"
#include "core/platform.h"

#include <algorithm>
#include <array>
#include <fstream>
#include <nlohmann/json.hpp>

namespace hybridai::io {
namespace {

Status ParseDType(const std::string& value, DType* dtype) {
    if (value == "F32") *dtype = DType::FP32;
    else if (value == "F16") *dtype = DType::FP16;
    else if (value == "BF16") *dtype = DType::BF16;
    else if (value == "F8_E4M3" || value == "F8_E4M3FN") {
        *dtype = DType::FP8_E4M3;
    } else if (value == "F8_E5M2") {
        *dtype = DType::FP8_E5M2;
    } else if (value == "I8") *dtype = DType::INT8;
    else if (value == "I32") *dtype = DType::INT32;
    else if (value == "I64") *dtype = DType::INT64;
    else if (value == "U8") *dtype = DType::UINT8;
    else if (value == "BOOL") *dtype = DType::BOOL;
    else {
        return Status(StatusCode::UnsupportedDType,
                      "Unsupported safetensors dtype: " + value);
    }
    return Status::OK();
}

bool ReadU64LittleEndian(std::istream& stream, uint64_t* value) {
    std::array<unsigned char, 8> bytes{};
    stream.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!stream) return false;

    uint64_t result = 0;
    for (size_t i = 0; i < bytes.size(); ++i) {
        result |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    *value = result;
    return true;
}

} // namespace

Status SafeTensorLoader::open(const std::string& path) {
    path_.clear();
    data_section_offset_ = 0;
    tensors_.clear();

    if (!platform::file_exists(path)) {
        return Status(StatusCode::FileNotFound,
                      "Safetensors file does not exist: " + path);
    }

    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return Status(StatusCode::FileNotFound,
                      "Failed to open safetensors file: " + path);
    }

    uint64_t header_size = 0;
    if (!ReadU64LittleEndian(file, &header_size)) {
        return Status(StatusCode::InvalidModel,
                      "Safetensors file is missing its header length");
    }

    const int64_t total_size = platform::file_size(path);
    if (total_size < 8 || header_size > static_cast<uint64_t>(total_size - 8)) {
        return Status(StatusCode::InvalidModel,
                      "Safetensors header length exceeds file size");
    }

    std::string header(static_cast<size_t>(header_size), '\0');
    file.read(header.data(), static_cast<std::streamsize>(header.size()));
    if (!file) {
        return Status(StatusCode::InvalidModel,
                      "Failed to read safetensors header");
    }

    nlohmann::json metadata;
    try {
        metadata = nlohmann::json::parse(header);
    } catch (const nlohmann::json::exception& error) {
        return Status(StatusCode::InvalidModel,
                      std::string("Invalid safetensors JSON header: ") +
                          error.what());
    }

    const uint64_t data_size = static_cast<uint64_t>(total_size) - 8 - header_size;
    for (const auto& [name, entry] : metadata.items()) {
        if (name == "__metadata__") continue;
        if (!entry.is_object() || !entry.contains("dtype") ||
            !entry.contains("shape") || !entry.contains("data_offsets")) {
            return Status(StatusCode::InvalidModel,
                          "Invalid tensor metadata for: " + name);
        }

        SafeTensorInfo info;
        Status dtype_status = ParseDType(entry.at("dtype").get<std::string>(),
                                         &info.dtype);
        if (!dtype_status.ok()) return dtype_status;

        std::vector<int64_t> dims;
        try {
            dims = entry.at("shape").get<std::vector<int64_t>>();
            const auto offsets =
                entry.at("data_offsets").get<std::vector<uint64_t>>();
            if (offsets.size() != 2 || offsets[0] > offsets[1] ||
                offsets[1] > data_size) {
                return Status(StatusCode::InvalidModel,
                              "Invalid data offsets for tensor: " + name);
            }
            info.data_begin = offsets[0];
            info.data_end = offsets[1];
        } catch (const nlohmann::json::exception& error) {
            return Status(StatusCode::InvalidModel,
                          std::string("Invalid tensor fields for ") + name +
                              ": " + error.what());
        }

        for (int64_t dim : dims) {
            if (dim < 0) {
                return Status(StatusCode::InvalidModel,
                              "Negative tensor dimension for: " + name);
            }
        }
        info.shape = Shape(std::move(dims));

        const uint64_t expected_bytes =
            static_cast<uint64_t>(info.shape.numel()) * SizeOfDType(info.dtype);
        if (expected_bytes != info.data_end - info.data_begin) {
            return Status(StatusCode::InvalidModel,
                          "Tensor byte size does not match shape for: " + name);
        }

        tensors_.emplace(name, std::move(info));
    }

    path_ = path;
    data_section_offset_ = 8 + header_size;
    return Status::OK();
}

std::vector<std::string> SafeTensorLoader::tensor_names() const {
    std::vector<std::string> names;
    names.reserve(tensors_.size());
    for (const auto& [name, info] : tensors_) {
        (void)info;
        names.push_back(name);
    }
    std::sort(names.begin(), names.end());
    return names;
}

const SafeTensorInfo* SafeTensorLoader::tensor_info(
    const std::string& name) const {
    const auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

Status SafeTensorLoader::load(const std::string& name, Device device,
                              Tensor* output,
                              MemoryType memory_type) const {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "SafeTensorLoader output must not be null");
    }
    const SafeTensorInfo* info = tensor_info(name);
    if (info == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "Tensor not found in safetensors file: " + name);
    }

    auto backend = BackendRegistry::instance().create_backend(device);
    if (backend == nullptr) {
        return Status(StatusCode::InvalidDevice,
                      "No backend available for requested tensor device");
    }

    const size_t bytes = static_cast<size_t>(info->data_end - info->data_begin);
    auto buffer = backend->create_buffer(bytes, memory_type);
    if (buffer == nullptr) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate tensor buffer");
    }

    std::ifstream file(path_, std::ios::binary);
    if (!file) {
        return Status(StatusCode::FileNotFound,
                      "Failed to reopen safetensors file: " + path_);
    }
    file.seekg(static_cast<std::streamoff>(data_section_offset_ +
                                           info->data_begin));
    if (!file) {
        return Status(StatusCode::InvalidModel,
                      "Failed to seek to tensor data: " + name);
    }

    std::vector<uint8_t> host_data(bytes);
    if (bytes > 0) {
        file.read(reinterpret_cast<char*>(host_data.data()),
                  static_cast<std::streamsize>(bytes));
    }
    if (!file && bytes > 0) {
        return Status(StatusCode::InvalidModel,
                      "Failed to read tensor data: " + name);
    }

    Status copy_status = backend->memcpy_h2d(buffer.get(), host_data.data(),
                                             bytes);
    if (!copy_status.ok()) return copy_status;

    *output = Tensor(info->shape, info->dtype, device, std::move(buffer));
    return Status::OK();
}

} // namespace hybridai::io
