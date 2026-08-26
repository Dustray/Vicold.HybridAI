#pragma once

#include "core/device.h"

#include <vector>

namespace hybridai {

class DeviceManager {
public:
    static DeviceManager& instance();

    // Initialize available devices based on compiled backends
    void initialize();

    const std::vector<Device>& devices() const noexcept { return devices_; }
    Device default_device() const;

    // Add a device manually (mainly for tests)
    void add_device(Device device);

private:
    DeviceManager() = default;

    std::vector<Device> devices_;
    bool initialized_ = false;
};

} // namespace hybridai
