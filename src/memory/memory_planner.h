#pragma once

#include "backends/interface/backend.h"
#include "core/device.h"

#include <cstddef>
#include <cstdint>

namespace hybridai {
namespace memory {

// Decides which allocator/memory type to use for a given device and workload.
//
// The policy is intentionally simple for Phase 2:
//   - CPU device -> Host memory
//   - Integrated GPU -> Unified memory (zero-copy with host)
//   - Discrete GPU -> Device memory
//   - If the requested model size exceeds available device memory, fall back to
//     Unified memory on iGPU or Host memory on dGPU.
//
// Future work: query actual device properties and include CPU offload tier.
class MemoryPlanner {
public:
    struct Policy {
        MemoryType memory_type = MemoryType::Host;
        // Suggested maximum resident bytes on the selected device before
        // offloading to a slower tier.
        size_t resident_limit_bytes = 0;
        bool allow_offload = false;
    };

    // Select allocator policy. `model_bytes` is the estimated total model
    // weight size; `kv_bytes` is the estimated KV cache size (0 if unknown).
    static Policy select_policy(const Device& device, size_t model_bytes,
                                size_t kv_bytes = 0);

    // Convenience: pick memory type only.
    static MemoryType select_memory_type(const Device& device,
                                         size_t model_bytes);
};

} // namespace memory
} // namespace hybridai
