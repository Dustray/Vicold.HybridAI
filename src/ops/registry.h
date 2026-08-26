#pragma once

#include "backends/interface/backend.h"
#include "core/device.h"
#include "core/dtype.h"
#include "core/status.h"
#include "core/tensor.h"

#include <functional>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>

namespace hybridai {
namespace ops {

// A kernel is an opaque callable that executes an operator given input tensors
// and produces output tensors. For Phase 3 this is a placeholder signature;
// concrete ops will expand the argument list and attributes.
using KernelFn = std::function<Status(const std::vector<Tensor*>& inputs,
                                      std::vector<Tensor*>& outputs)>;

// Key for kernel lookup.
struct KernelKey {
    std::string op_name;
    DeviceType device_type;
    DType dtype;

    bool operator==(const KernelKey& other) const noexcept {
        return op_name == other.op_name && device_type == other.device_type &&
               dtype == other.dtype;
    }
};

} // namespace ops
} // namespace hybridai

namespace std {
template <>
struct hash<hybridai::ops::KernelKey> {
    size_t operator()(const hybridai::ops::KernelKey& key) const noexcept {
        size_t h1 = std::hash<std::string>{}(key.op_name);
        size_t h2 = std::hash<int>{}(static_cast<int>(key.device_type));
        size_t h3 = std::hash<int>{}(static_cast<int>(key.dtype));
        return h1 ^ (h2 << 1) ^ (h3 << 2);
    }
};
} // namespace std

namespace hybridai {
namespace ops {

// Central registry for operator kernels. Backends and plugin modules can
// register custom kernels at load time. The executor queries the registry to
// select a kernel based on (op_name, device_type, dtype).
class KernelRegistry {
public:
    static KernelRegistry& instance();

    // Register a kernel. Returns false if a kernel for the same key already
    // exists and overwrite is false.
    bool register_kernel(const KernelKey& key, KernelFn kernel,
                         bool overwrite = false);

    // Find a kernel. Returns nullptr if not found.
    KernelFn find_kernel(const KernelKey& key) const;

    // Check existence.
    bool has_kernel(const KernelKey& key) const;

    // Remove a kernel (mainly for tests).
    bool unregister_kernel(const KernelKey& key);

    size_t kernel_count() const;

private:
    KernelRegistry() = default;

    mutable std::mutex mutex_;
    std::unordered_map<KernelKey, KernelFn> kernels_;
};

class KernelRegistrar {
public:
    KernelRegistrar(const KernelKey& key, KernelFn kernel) {
        KernelRegistry::instance().register_kernel(key, kernel);
    }
};

#define HYBRIDAI_REGISTER_KERNEL(op_name, device_type, dtype, fn)          \
    static ::hybridai::ops::KernelRegistrar g_hybridai_kernel_##op_name##_##device_type( \
        ::hybridai::ops::KernelKey{#op_name,                                \
                                    ::hybridai::DeviceType::device_type,     \
                                    ::hybridai::DType::dtype},               \
        fn)

} // namespace ops
} // namespace hybridai
