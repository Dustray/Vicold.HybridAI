#include "memory/memory_planner.h"

namespace hybridai {
namespace memory {

MemoryPlanner::Policy MemoryPlanner::select_policy(const Device& device,
                                                   size_t model_bytes,
                                                   size_t kv_bytes) {
    Policy policy;
    policy.allow_offload = false;

    if (device.is_cpu()) {
        policy.memory_type = MemoryType::Host;
        policy.resident_limit_bytes = 0;
        return policy;
    }

    // Placeholder device memory limits. These should be replaced by runtime
    // queries once Backend exposes device memory info.
    constexpr size_t kIgpuTotalBytes = 13ULL * 1024 * 1024 * 1024; // 13 GiB
    constexpr size_t kDgpuTotalBytes = 8ULL * 1024 * 1024 * 1024;  // 8 GiB
    constexpr size_t kSafetyHeadroom = 512ULL * 1024 * 1024;       // 512 MiB

    size_t total_bytes = device.type() == DeviceType::IntegratedGPU
                           ? kIgpuTotalBytes
                           : kDgpuTotalBytes;
    size_t required = model_bytes + kv_bytes + kSafetyHeadroom;

    if (device.type() == DeviceType::IntegratedGPU) {
        if (required <= total_bytes && device.unified_memory_supported()) {
            policy.memory_type = MemoryType::Unified;
        } else {
            // Unified memory still works even if it overflows to host.
            policy.memory_type = MemoryType::Unified;
            policy.allow_offload = true;
        }
        policy.resident_limit_bytes = total_bytes - kSafetyHeadroom;
    } else {
        // Discrete GPU
        if (required <= total_bytes) {
            policy.memory_type = MemoryType::Device;
            policy.resident_limit_bytes = total_bytes - kSafetyHeadroom;
        } else {
            // Not enough VRAM: fall back to host memory (CPU offload path).
            policy.memory_type = MemoryType::Host;
            policy.allow_offload = true;
            policy.resident_limit_bytes = 0;
        }
    }

    return policy;
}

MemoryType MemoryPlanner::select_memory_type(const Device& device,
                                             size_t model_bytes) {
    return select_policy(device, model_bytes, 0).memory_type;
}

} // namespace memory
} // namespace hybridai
