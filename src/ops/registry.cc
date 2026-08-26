#include "ops/registry.h"

namespace hybridai {
namespace ops {

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry registry;
    return registry;
}

bool KernelRegistry::register_kernel(const KernelKey& key, KernelFn kernel,
                                     bool overwrite) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernels_.find(key);
    if (it != kernels_.end() && !overwrite) {
        return false;
    }
    kernels_[key] = std::move(kernel);
    return true;
}

KernelFn KernelRegistry::find_kernel(const KernelKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = kernels_.find(key);
    if (it == kernels_.end()) {
        return nullptr;
    }
    return it->second;
}

bool KernelRegistry::has_kernel(const KernelKey& key) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.find(key) != kernels_.end();
}

bool KernelRegistry::unregister_kernel(const KernelKey& key) {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.erase(key) > 0;
}

size_t KernelRegistry::kernel_count() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return kernels_.size();
}

} // namespace ops
} // namespace hybridai
