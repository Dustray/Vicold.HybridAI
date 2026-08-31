#pragma once

#include "backends/interface/backend.h"

#include <cstddef>
#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <mutex>

namespace hybridai {
namespace memory {

// A bucket-based caching allocator sitting on top of a backend Allocator.
//
// - Allocations are rounded up to bucket sizes (powers of two, optionally
//   with a linear sub-range for small blocks).
// - Freed blocks are returned to a per-bucket free list for reuse.
// - Optional high-water limit triggers eager release back to the underlying
//   allocator when exceeded.
//
// Thread safety: all public methods are internally synchronized.
class MemoryPool final : public Allocator {
public:
    struct Options {
        // Smallest allocation size handled by the pool.
        size_t min_bucket_size = 256;
        // Largest allocation size handled by the pool; larger allocations go
        // straight to the underlying allocator.
        size_t max_bucket_size = 64 * 1024 * 1024; // 64 MiB
        // Growth factor between buckets (2.0 = power-of-two).
        double bucket_growth = 2.0;
        // Maximum total bytes retained by the pool (0 = unlimited).
        size_t max_pooled_bytes = 256 * 1024 * 1024; // 256 MiB
    };

    MemoryPool(Backend* backend, MemoryType type);
    MemoryPool(Backend* backend, MemoryType type, const Options& options);
    ~MemoryPool() override;

    // Allocator interface
    Status allocate(size_t size, void** ptr) override;
    Status deallocate(void* ptr) override;
    MemoryType memory_type() const noexcept override { return type_; }

    // Statistics
    size_t pooled_bytes() const noexcept { return pooled_bytes_; }
    size_t active_bytes() const noexcept { return active_bytes_; }
    size_t total_allocations() const noexcept { return total_allocs_; }
    size_t cache_hits() const noexcept { return cache_hits_; }

    // Release all cached blocks back to the backend allocator.
    Status release_cache();

private:
    size_t bucket_index(size_t size) const;

    struct Block {
        void* ptr = nullptr;
        size_t size = 0;
        bool active = false;
    };

    struct ActiveBlock {
        size_t bucket = 0;
        size_t size = 0;
    };

    std::unique_ptr<Allocator> allocator_;
    MemoryType type_;
    Options options_;

    std::vector<size_t> bucket_sizes_;
    // bucket_index -> list of free blocks
    std::map<size_t, std::list<Block>> free_blocks_;
    // ptr -> allocation metadata
    std::map<void*, ActiveBlock> active_blocks_;

    mutable std::mutex mutex_;
    size_t pooled_bytes_ = 0;
    size_t active_bytes_ = 0;
    size_t total_allocs_ = 0;
    size_t cache_hits_ = 0;
};

} // namespace memory
} // namespace hybridai
