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

    // Enumerate devices for every registered backend. Each backend reports its
    // own native ids (which may already be filtered by environment variables such
    // as HIP_VISIBLE_DEVICES), type, and unified-memory capability.
    for (const std::string& backend_name :
         BackendRegistry::instance().backend_names()) {
        // CPU backend is added explicitly above.
        if (backend_name == "cpu") {
            continue;
        }

        Device probe_device(-1, DeviceType::CPU, backend_name, false);
        auto probe = BackendRegistry::instance().create_backend(probe_device);
        if (probe == nullptr) {
            continue;
        }

        for (const Device& device : probe->enumerate_devices()) {
            auto backend = BackendRegistry::instance().create_backend(device);
            if (backend != nullptr && backend->is_available()) {
                devices_.push_back(device);
            }
        }
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
