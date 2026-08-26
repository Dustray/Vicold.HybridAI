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
        // TODO: move native property discovery behind a backend enumeration API.
        // Until then, only publish configured devices that the backend confirms
        // are present. The type/unified-memory hints match the target platform.
        const Device hip_devices[] = {
            Device(0, DeviceType::DiscreteGPU, "hip", false),
            Device(1, DeviceType::IntegratedGPU, "hip", true),
        };
        for (const Device& device : hip_devices) {
            auto backend = BackendRegistry::instance().create_backend(device);
            if (backend != nullptr && backend->is_available()) {
                devices_.push_back(device);
            }
        }
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
    for (const Device& device : devices_) {
        if (device.is_gpu()) {
            return device;
        }
    }
    return devices_.front();
}

void DeviceManager::add_device(Device device) { devices_.push_back(device); }

} // namespace hybridai
