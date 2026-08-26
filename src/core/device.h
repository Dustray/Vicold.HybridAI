#pragma once

#include <cstdint>
#include <string>
#include <string_view>

namespace hybridai {

enum class DeviceType : uint8_t {
    CPU,
    IntegratedGPU,
    DiscreteGPU,
    Unknown,
};

class Device {
public:
    Device() noexcept : id_(0), type_(DeviceType::CPU) {}
    Device(int32_t id, DeviceType type, std::string_view backend,
           bool unified_memory = false)
        : id_(id), type_(type), backend_(backend),
          unified_memory_supported_(unified_memory) {}

    int32_t id() const noexcept { return id_; }
    DeviceType type() const noexcept { return type_; }
    const std::string& backend() const noexcept { return backend_; }
    bool unified_memory_supported() const noexcept {
        return unified_memory_supported_;
    }

    bool is_cpu() const noexcept { return type_ == DeviceType::CPU; }
    bool is_gpu() const noexcept {
        return type_ == DeviceType::IntegratedGPU ||
               type_ == DeviceType::DiscreteGPU;
    }

    static Device Cpu() noexcept {
        return Device(0, DeviceType::CPU, "cpu", false);
    }

    bool operator==(const Device& other) const noexcept {
        return id_ == other.id_ && type_ == other.type_ &&
               backend_ == other.backend_;
    }
    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }

private:
    int32_t id_;
    DeviceType type_;
    std::string backend_;
    bool unified_memory_supported_ = false;
};

constexpr std::string_view DeviceTypeToString(DeviceType type) {
    switch (type) {
        case DeviceType::CPU: return "cpu";
        case DeviceType::IntegratedGPU: return "igpu";
        case DeviceType::DiscreteGPU: return "dgpu";
        case DeviceType::Unknown: return "unknown";
    }
    return "unknown";
}

} // namespace hybridai
