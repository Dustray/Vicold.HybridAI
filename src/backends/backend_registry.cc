#include "backends/backend_registry.h"

#include <algorithm>
#include <cstdint>

namespace hybridai {

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::register_backend(std::string_view name,
                                       BackendFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    const std::string backend_name(name);
    factories_[backend_name] = std::move(factory);
    const std::string prefix = backend_name + ":";
    for (auto it = backends_.begin(); it != backends_.end();) {
        if (it->first.starts_with(prefix)) {
            it = backends_.erase(it);
        } else {
            ++it;
        }
    }
}

std::unique_ptr<Backend> BackendRegistry::create_backend(const Device& device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device.backend().empty()) {
        return nullptr;
    }

    auto it = factories_.find(device.backend());
    if (it == factories_.end()) {
        return nullptr;
    }
    return it->second(device);
}

std::shared_ptr<Backend> BackendRegistry::get_backend(const Device& device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device.backend().empty()) return nullptr;
    const std::string key = device.backend() + ":" +
                            std::to_string(device.id()) + ":" +
                            std::to_string(static_cast<uint8_t>(device.type()));
    auto cached = backends_.find(key);
    if (cached != backends_.end()) return cached->second;

    auto factory = factories_.find(device.backend());
    if (factory == factories_.end()) return nullptr;
    std::shared_ptr<Backend> backend = factory->second(device);
    if (backend != nullptr) backends_.emplace(key, backend);
    return backend;
}

bool BackendRegistry::has_backend(std::string_view name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return factories_.find(std::string(name)) != factories_.end();
}

std::vector<std::string> BackendRegistry::backend_names() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> names;
    names.reserve(factories_.size());
    for (const auto& [name, _factory] : factories_) {
        names.push_back(name);
    }
    return names;
}

} // namespace hybridai
