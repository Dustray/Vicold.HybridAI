#include "memory/memory_pool.h"

#include <algorithm>
#include <cmath>

namespace hybridai {
namespace memory {

MemoryPool::MemoryPool(Backend* backend, MemoryType type)
    : MemoryPool(backend, type, Options{}) {}

MemoryPool::MemoryPool(Backend* backend, MemoryType type, const Options& options)
        : allocator_(backend == nullptr ? nullptr : backend->create_allocator(type)),
            type_(type), options_(options) {
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
    if (ptr == nullptr || allocator_ == nullptr) {
        return Status(StatusCode::InvalidArgument,
                      "MemoryPool: allocator or output pointer is null");
    }
    if (size == 0) {
        *ptr = nullptr;
        return Status::OK();
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Oversized allocations bypass the pool.
    if (size > options_.max_bucket_size) {
        Status s = allocator_->allocate(size, ptr);
        if (!s.ok()) return s;
        active_blocks_[*ptr] = {
            std::numeric_limits<size_t>::max(), size};
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
        active_blocks_[*ptr] = {bucket, block.size};
        active_bytes_ += block.size;
        pooled_bytes_ -= block.size;
        ++total_allocs_;
        ++cache_hits_;
        return Status::OK();
    }

    // Allocate a new block of the bucket size.
    size_t alloc_size = bucket_sizes_[bucket];
    Status s = allocator_->allocate(alloc_size, ptr);
    if (!s.ok()) return s;

    active_blocks_[*ptr] = {bucket, alloc_size};
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

    size_t bucket = it->second.bucket;
    size_t block_size = it->second.size;
    active_bytes_ -= block_size;
    if (bucket == std::numeric_limits<size_t>::max()) {
        // Direct allocation; release immediately.
        allocator_->deallocate(ptr);
        active_blocks_.erase(it);
        return Status::OK();
    }

    if (options_.max_pooled_bytes > 0 &&
        pooled_bytes_ + block_size > options_.max_pooled_bytes) {
        // Over high-water mark; release directly.
        allocator_->deallocate(ptr);
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

    for (auto& [bucket, blocks] : free_blocks_) {
        for (auto& block : blocks) {
            allocator_->deallocate(block.ptr);
        }
    }
    free_blocks_.clear();
    pooled_bytes_ = 0;
    return Status::OK();
}

size_t MemoryPool::bucket_index(size_t size) const {
    const size_t requested = std::max(size, options_.min_bucket_size);
    auto it =
        std::lower_bound(bucket_sizes_.begin(), bucket_sizes_.end(), requested);
    if (it == bucket_sizes_.end()) {
        return bucket_sizes_.size() - 1;
    }
    return static_cast<size_t>(std::distance(bucket_sizes_.begin(), it));
}

} // namespace memory
} // namespace hybridai
