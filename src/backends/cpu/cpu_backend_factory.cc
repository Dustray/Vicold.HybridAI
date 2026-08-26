#include "backends/backend_registry.h"
#include "backends/cpu/cpu_backend.h"

namespace hybridai {

namespace {

std::unique_ptr<Backend> create_cpu_backend(const Device& device) {
    return std::make_unique<CpuBackend>(device);
}

HYBRIDAI_REGISTER_BACKEND(cpu, create_cpu_backend);

} // namespace

void InitializeBuiltinBackends() {
    // The static registrar above performs the actual registration.
    // Calling this function forces the translation unit to be linked,
    // ensuring the registrar runs before main().
    (void)BackendRegistry::instance().has_backend("cpu");
}

} // namespace hybridai
