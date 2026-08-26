#include "core/device_manager.h"

#include "backends/backend_registry.h"

namespace hybridai {

DeviceManager& DeviceManager::instance() {
    static DeviceManager manager;
    return manager;
}

void DeviceManager::initialize() {
    if (initialized_) {
        return;
    }

    devices_.clear();

    // Always register CPU device
    devices_.emplace_back(Device::Cpu());

    if (BackendRegistry::instance().has_backend("hip")) {
        // TODO: query HIP device count and properties
        devices_.emplace_back(0, DeviceType::DiscreteGPU, "hip", false);
        devices_.emplace_back(1, DeviceType::IntegratedGPU, "hip", true);
    }

    if (BackendRegistry::instance().has_backend("cuda")) {
        // TODO: query CUDA device count and properties
        devices_.emplace_back(0, DeviceType::DiscreteGPU, "cuda", false);
    }

    initialized_ = true;
}

Device DeviceManager::default_device() const {
    if (devices_.empty()) {
        return Device::Cpu();
    }
    return devices_.front();
}

void DeviceManager::add_device(Device device) { devices_.push_back(device); }

} // namespace hybridai
