#include "backends/backend_registry.h"
#include "backends/cpu/cpu_backend.h"

namespace hybridai {

namespace {

std::unique_ptr<Backend> create_cpu_backend(const Device& device) {
    return std::make_unique<CpuBackend>(device);
}

HYBRIDAI_REGISTER_BACKEND(cpu, create_cpu_backend);

} // namespace

void InitializeHipBackend();
void InitializeCpuBackend();

void InitializeBuiltinBackends() {
    // Calling the per-backend helpers forces each translation unit containing
    // the static BackendRegistrar to be linked, which performs the actual
    // registration before main().
    InitializeCpuBackend();
    InitializeHipBackend();
}

void InitializeCpuBackend() {
    (void)BackendRegistry::instance().has_backend("cpu");
}

} // namespace hybridai
