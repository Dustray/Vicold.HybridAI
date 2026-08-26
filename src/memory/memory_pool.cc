#include "memory/memory_pool.h"

#include <algorithm>
#include <cmath>

namespace hybridai {
namespace memory {

namespace {

size_t next_power_of_two(size_t v) {
    if (v == 0) return 1;
    --v;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    if constexpr (sizeof(size_t) == 8) {
        v |= v >> 32;
    }
    return v + 1;
}

} // namespace

MemoryPool::MemoryPool(Backend* backend, MemoryType type, const Options& options)
    : backend_(backend), type_(type), options_(options) {
    if (options_.bucket_growth < 1.01) {
        options_.bucket_growth = 2.0;
    }

    size_t size = options_.min_bucket_size;
    while (size <= options_.max_bucket_size) {
        bucket_sizes_.push_back(size);
        size = static_cast<size_t>(std::ceil(size * options_.bucket_growth));
        // Ensure strict growth to avoid an infinite loop with growth == 1.0.
        if (size <= bucket_sizes_.back()) {
            size = bucket_sizes_.back() * 2;
        }
    }
    if (bucket_sizes_.empty() || bucket_sizes_.back() < options_.max_bucket_size) {
        bucket_sizes_.push_back(options_.max_bucket_size);
    }
}

MemoryPool::~MemoryPool() {
    release_cache();
}

Status MemoryPool::allocate(size_t size, void** ptr) {
    if (size == 0) {
        *ptr = nullptr;
        return Status::OK();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Oversized allocations bypass the pool.
    if (size > options_.max_bucket_size) {
        auto allocator = backend_->create_allocator(type_);
        Status s = allocator->allocate(size, ptr);
        if (!s.ok()) return s;
        active_blocks_[*ptr] = {std::numeric_limits<size_t>::max(), {}};
        active_bytes_ += size;
        ++total_allocs_;
        return Status::OK();
    }

    size_t bucket = bucket_index(size);
    auto it = free_blocks_.find(bucket);
    if (it != free_blocks_.end() && !it->second.empty()) {
        // Reuse a cached block.
        auto block_it = it->second.begin();
        Block block = std::move(*block_it);
        it->second.erase(block_it);
        block.active = true;
        *ptr = block.ptr;
        active_blocks_[*ptr] = {bucket, {}};
        active_bytes_ += block.size;
        pooled_bytes_ -= block.size;
        ++total_allocs_;
        ++cache_hits_;
        return Status::OK();
    }

    // Allocate a new block of the bucket size.
    size_t alloc_size = bucket_sizes_[bucket];
    auto allocator = backend_->create_allocator(type_);
    Status s = allocator->allocate(alloc_size, ptr);
    if (!s.ok()) return s;

    active_blocks_[*ptr] = {bucket, {}};
    active_bytes_ += alloc_size;
    ++total_allocs_;
    return Status::OK();
}

Status MemoryPool::deallocate(void* ptr) {
    if (ptr == nullptr) {
        return Status::OK();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = active_blocks_.find(ptr);
    if (it == active_blocks_.end()) {
        return Status(StatusCode::InvalidArgument,
                      "MemoryPool: deallocating unknown pointer");
    }

    size_t bucket = it->second.first;
    if (bucket == std::numeric_limits<size_t>::max()) {
        // Direct allocation; release immediately.
        auto allocator = backend_->create_allocator(type_);
        allocator->deallocate(ptr);
        active_blocks_.erase(it);
        return Status::OK();
    }

    size_t block_size = bucket_sizes_[bucket];
    active_bytes_ -= block_size;

    if (options_.max_pooled_bytes > 0 &&
        pooled_bytes_ + block_size > options_.max_pooled_bytes) {
        // Over high-water mark; release directly.
        auto allocator = backend_->create_allocator(type_);
        allocator->deallocate(ptr);
        active_blocks_.erase(it);
        return Status::OK();
    }

    Block block;
    block.ptr = ptr;
    block.size = block_size;
    block.active = false;
    free_blocks_[bucket].push_back(block);
    pooled_bytes_ += block_size;
    active_blocks_.erase(it);
    return Status::OK();
}

Status MemoryPool::release_cache() {
    std::lock_guard<std::mutex> lock(mutex_);

    auto allocator = backend_->create_allocator(type_);
    for (auto& [bucket, blocks] : free_blocks_) {
        for (auto& block : blocks) {
            allocator->deallocate(block.ptr);
        }
    }
    free_blocks_.clear();
    pooled_bytes_ = 0;
    return Status::OK();
}

size_t MemoryPool::bucket_index(size_t size) const {
    size_t rounded = next_power_of_two(size);
    if (rounded < options_.min_bucket_size) {
        rounded = options_.min_bucket_size;
    }
    auto it = std::lower_bound(bucket_sizes_.begin(), bucket_sizes_.end(), rounded);
    if (it == bucket_sizes_.end()) {
        return bucket_sizes_.size() - 1;
    }
    return static_cast<size_t>(std::distance(bucket_sizes_.begin(), it));
}

} // namespace memory
} // namespace hybridai
