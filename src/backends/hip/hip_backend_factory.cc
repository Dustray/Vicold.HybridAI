#include "backends/backend_registry.h"
#include "backends/hip/hip_backend.h"

namespace hybridai {

namespace {

std::unique_ptr<Backend> create_hip_backend(const Device& device) {
    return std::make_unique<HipBackend>(device);
}

HYBRIDAI_REGISTER_BACKEND(hip, create_hip_backend);

} // namespace

void InitializeHipBackend() {
    // Forces this translation unit to be linked so the static registrar runs.
    (void)BackendRegistry::instance().has_backend("hip");
}

} // namespace hybridai
