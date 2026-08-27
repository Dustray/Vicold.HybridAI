#pragma once

#include "backends/interface/backend.h"

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>

namespace hybridai {

class BackendRegistry {
public:
    static BackendRegistry& instance();

    void register_backend(std::string_view name, BackendFactory factory);
    std::unique_ptr<Backend> create_backend(const Device& device);

    bool has_backend(std::string_view name) const;

    // Return the names of all currently registered backends. Order matches
    // registration order so callers can prioritise preferred backends.
    std::vector<std::string> backend_names() const;

private:
    BackendRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<std::string, BackendFactory> factories_;
};

// Explicitly initialize all built-in backends. Call this in executables
// (CLI, tests, benchmarks) to ensure backend factories are registered even when
// linking HybridAI as a static library. Static-library global constructors may
// be elided by the linker on some platforms/toolchains.
void InitializeBuiltinBackends();

class BackendRegistrar {
public:
    explicit BackendRegistrar(std::string_view name, BackendFactory factory) {
        BackendRegistry::instance().register_backend(name, std::move(factory));
    }
};

#define HYBRIDAI_REGISTER_BACKEND(name, factory) \
    static ::hybridai::BackendRegistrar g_hybridai_backend_##name( \
        #name, factory)

} // namespace hybridai
