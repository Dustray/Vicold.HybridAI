#include "io/safetensor_loader.h"

#include "backends/backend_registry.h"
#include "core/platform.h"

#include <algorithm>
#include <array>
#include <cstring>
#include <fstream>
#include <filesystem>
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
    tensor_shards_.clear();
    shards_.clear();

    if (!platform::file_exists(path)) {
        return Status(StatusCode::FileNotFound,
                      "Safetensors file does not exist: " + path);
    }

    Shard shard;
    Status status = parse_shard(path, &shard);
    if (!status.ok()) return status;

    path_ = path;
    data_section_offset_ = shard.data_section_offset;
    tensors_ = std::move(shard.tensors);
    return Status::OK();
}

Status SafeTensorLoader::parse_shard(const std::string& path, Shard* shard) {
    if (shard == nullptr) {
        return Status(StatusCode::InvalidArgument, "Shard must not be null");
    }
    shard->path.clear();
    shard->data_section_offset = 0;
    shard->tensors.clear();

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

            shard->tensors.emplace(name, std::move(info));
    }

        shard->path = path;
        shard->data_section_offset = 8 + header_size;
    return Status::OK();
}

Status SafeTensorLoader::open_model(const std::string& path) {
    if (!std::filesystem::is_directory(path)) {
        return open(path);
    }

    const auto index_path = std::filesystem::path(path) /
                            "model.safetensors.index.json";
    if (!platform::file_exists(index_path.string())) {
        const auto single_file = std::filesystem::path(path) /
                                 "model.safetensors";
        if (platform::file_exists(single_file.string())) {
            return open(single_file.string());
        }
        return Status(StatusCode::FileNotFound,
                      "Neither safetensors index nor model.safetensors exists: " +
                          path);
    }

    std::ifstream index_file(index_path);
    if (!index_file) {
        return Status(StatusCode::FileNotFound,
                      "Failed to open safetensors index: " +
                          index_path.string());
    }

    nlohmann::json index;
    try {
        index_file >> index;
    } catch (const nlohmann::json::exception& error) {
        return Status(StatusCode::InvalidModel,
                      std::string("Invalid safetensors index JSON: ") +
                          error.what());
    }
    if (!index.contains("weight_map") || !index["weight_map"].is_object()) {
        return Status(StatusCode::InvalidModel,
                      "Safetensors index has no valid weight_map");
    }

    path_.clear();
    data_section_offset_ = 0;
    tensors_.clear();
    tensor_shards_.clear();
    shards_.clear();

    for (const auto& [name, shard_name_json] : index["weight_map"].items()) {
        if (!shard_name_json.is_string()) {
            return Status(StatusCode::InvalidModel,
                          "Invalid shard name for tensor: " + name);
        }
        const std::string shard_name = shard_name_json.get<std::string>();
        const std::string shard_path =
            (std::filesystem::path(path) / shard_name).string();
        if (shards_.find(shard_name) == shards_.end()) {
            Shard shard;
            Status status = parse_shard(shard_path, &shard);
            if (!status.ok()) return status;
            shards_.emplace(shard_name, std::move(shard));
        }
        const auto& shard = shards_.at(shard_name);
        const auto tensor_it = shard.tensors.find(name);
        if (tensor_it == shard.tensors.end()) {
            return Status(StatusCode::InvalidModel,
                          "Tensor listed in index is missing from shard: " +
                              name);
        }
        tensors_.emplace(name, tensor_it->second);
        tensor_shards_.emplace(name, shard_name);
    }

    path_ = path;
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

    std::string source_path = path_;
    uint64_t source_data_offset = data_section_offset_;
    if (!tensor_shards_.empty()) {
        const auto shard_name_it = tensor_shards_.find(name);
        if (shard_name_it == tensor_shards_.end()) {
            return Status(StatusCode::InvalidModel,
                          "No shard mapping for tensor: " + name);
        }
        const auto shard_it = shards_.find(shard_name_it->second);
        if (shard_it == shards_.end()) {
            return Status(StatusCode::InvalidModel,
                          "Mapped shard is not loaded: " +
                              shard_name_it->second);
        }
        source_path = shard_it->second.path;
        source_data_offset = shard_it->second.data_section_offset;
    }

    std::ifstream file(source_path, std::ios::binary);
    if (!file) {
        return Status(StatusCode::FileNotFound,
                      "Failed to reopen safetensors file: " + source_path);
    }
    file.seekg(static_cast<std::streamoff>(source_data_offset +
                                           info->data_begin));
    if (!file) {
        return Status(StatusCode::InvalidModel,
                      "Failed to seek to tensor data: " + name);
    }

    // Use pinned host staging for HIP asynchronously and ordinary host memory
    // for CPU.  The staging buffer is released immediately after synchronize,
    // keeping peak RSS lower while loading a large sharded model.
    std::unique_ptr<Allocator> staging_allocator =
        backend->create_allocator(device.is_cpu() ? MemoryType::Host
                                                  : MemoryType::HostPinned);
    void* staging = nullptr;
    Status staging_status = staging_allocator->allocate(bytes, &staging);
    if (!staging_status.ok()) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate safetensors staging buffer: " + name);
    }
    if (bytes > 0) {
        file.read(static_cast<char*>(staging),
                  static_cast<std::streamsize>(bytes));
    }
    if (!file && bytes > 0) {
        staging_allocator->deallocate(staging);
        return Status(StatusCode::InvalidModel,
                      "Failed to read tensor data: " + name);
    }

    Status copy_status = backend->memcpy_h2d(buffer.get(), staging, bytes);
    if (!copy_status.ok()) {
        staging_allocator->deallocate(staging);
        return copy_status;
    }
    // The temporary host buffer must remain alive until an asynchronous HIP
    // copy has completed. This is also a no-op-style synchronization for CPU.
    copy_status = backend->synchronize();
    staging_allocator->deallocate(staging);
    if (!copy_status.ok()) return copy_status;

    *output = Tensor(info->shape, info->dtype, device, std::move(buffer));
    return Status::OK();
}

Status SafeTensorLoader::load_as_fp32(const std::string& name, Device device,
                                      Tensor* output,
                                      MemoryType memory_type) const {
    if (output == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "SafeTensorLoader output must not be null");
    }
    const SafeTensorInfo* info = tensor_info(name);
    if (info == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "Tensor not found in safetensors model: " + name);
    }
    if (info->dtype != DType::BF16 && info->dtype != DType::FP32) {
        return Status(StatusCode::UnsupportedDType,
                      "Only BF16 and FP32 can be loaded as FP32");
    }

    Tensor source;
    Status status = load(name, device, &source, memory_type);
    if (!status.ok()) return status;
    if (info->dtype == DType::FP32) {
        *output = std::move(source);
        return Status::OK();
    }

    auto backend = BackendRegistry::instance().create_backend(device);
    if (backend == nullptr) {
        return Status(StatusCode::InvalidDevice,
                      "No backend available for FP32 conversion");
    }
    auto buffer = backend->create_buffer(source.numel() * sizeof(float),
                                         memory_type);
    if (buffer == nullptr) {
        return Status(StatusCode::OutOfMemory,
                      "Failed to allocate FP32 scale tensor");
    }
    const auto* input = static_cast<const uint16_t*>(source.data());
    auto* values = static_cast<float*>(buffer->data());
    for (int64_t i = 0; i < source.numel(); ++i) {
        const uint32_t bits = static_cast<uint32_t>(input[i]) << 16u;
        float value;
        std::memcpy(&value, &bits, sizeof(value));
        values[i] = value;
    }
    *output = Tensor(source.shape(), DType::FP32, device, std::move(buffer));
    return Status::OK();
}

} // namespace hybridai::io
