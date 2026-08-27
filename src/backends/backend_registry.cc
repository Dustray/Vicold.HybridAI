#include "backends/backend_registry.h"

#include <algorithm>

namespace hybridai {

BackendRegistry& BackendRegistry::instance() {
    static BackendRegistry registry;
    return registry;
}

void BackendRegistry::register_backend(std::string_view name,
                                       BackendFactory factory) {
    std::lock_guard<std::mutex> lock(mutex_);
    factories_[std::string(name)] = std::move(factory);
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
