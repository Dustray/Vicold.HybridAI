#include "backends/backend_registry.h"
#include "backends/cpu/cpu_backend.h"

namespace hybridai {

namespace {

std::unique_ptr<Backend> create_cpu_backend(const Device& device) {
    return std::make_unique<CpuBackend>(device);
}

HYBRIDAI_REGISTER_BACKEND(cpu, create_cpu_backend);

} // namespace

} // namespace hybridai
