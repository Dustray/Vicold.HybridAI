#include "backends/backend_registry.h"
#include "core/device.h"
#include "memory/memory_planner.h"
#include "memory/memory_pool.h"

#include <gtest/gtest.h>

#include <cstring>

using namespace hybridai;
using namespace hybridai::memory;

TEST(MemoryTest, PoolAllocateDeallocate) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    ASSERT_NE(backend, nullptr);

    MemoryPool pool(backend.get(), MemoryType::Host);
    void* p = nullptr;
    EXPECT_TRUE(pool.allocate(1024, &p).ok());
    EXPECT_NE(p, nullptr);

    // Write and read back to ensure the allocation is usable.
    std::memset(p, 0xAB, 1024);
    EXPECT_EQ(static_cast<uint8_t*>(p)[0], 0xAB);

    EXPECT_TRUE(pool.deallocate(p).ok());
}

TEST(MemoryTest, PoolCacheReuse) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    ASSERT_NE(backend, nullptr);

    MemoryPool pool(backend.get(), MemoryType::Host);
    void* p1 = nullptr;
    void* p2 = nullptr;
    EXPECT_TRUE(pool.allocate(1000, &p1).ok());
    EXPECT_TRUE(pool.deallocate(p1).ok());
    EXPECT_TRUE(pool.allocate(1000, &p2).ok());

    // With power-of-two buckets the same address should be reused.
    EXPECT_EQ(p1, p2);
    EXPECT_EQ(pool.cache_hits(), 1u);

    EXPECT_TRUE(pool.deallocate(p2).ok());
}

TEST(MemoryTest, PoolHighWaterRelease) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    ASSERT_NE(backend, nullptr);

    MemoryPool::Options options;
    options.min_bucket_size = 64;
    options.max_bucket_size = 1024;
    options.max_pooled_bytes = 256; // Very small cache
    MemoryPool pool(backend.get(), MemoryType::Host, options);

    void* p = nullptr;
    EXPECT_TRUE(pool.allocate(512, &p).ok());
    EXPECT_EQ(pool.pooled_bytes(), 0u);
    EXPECT_TRUE(pool.deallocate(p).ok());
    // 512-byte block exceeds 256-byte high-water; should be released.
    EXPECT_EQ(pool.pooled_bytes(), 0u);
}

TEST(MemoryTest, PoolOversizedDirectAllocation) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    ASSERT_NE(backend, nullptr);

    MemoryPool::Options options;
    options.max_bucket_size = 1024;
    MemoryPool pool(backend.get(), MemoryType::Host, options);

    void* p = nullptr;
    EXPECT_TRUE(pool.allocate(2048, &p).ok());
    EXPECT_NE(p, nullptr);
    EXPECT_TRUE(pool.deallocate(p).ok());
}

TEST(MemoryTest, PlannerCpuDevice) {
    Device cpu = Device::Cpu();
    auto policy = MemoryPlanner::select_policy(cpu, 1024);
    EXPECT_EQ(policy.memory_type, MemoryType::Host);
    EXPECT_FALSE(policy.allow_offload);
}

TEST(MemoryTest, PlannerIntegratedGpu) {
    Device igpu(0, DeviceType::IntegratedGPU, "hip", true);
    auto policy = MemoryPlanner::select_policy(igpu, 4ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(policy.memory_type, MemoryType::Unified);
    EXPECT_TRUE(policy.allow_offload || policy.resident_limit_bytes > 0);
}

TEST(MemoryTest, PlannerDiscreteGpuFits) {
    Device dgpu(0, DeviceType::DiscreteGPU, "hip", false);
    auto policy = MemoryPlanner::select_policy(dgpu, 4ULL * 1024 * 1024 * 1024);
    EXPECT_EQ(policy.memory_type, MemoryType::Device);
}

TEST(MemoryTest, PlannerDiscreteGpuOverflow) {
    Device dgpu(0, DeviceType::DiscreteGPU, "hip", false);
    auto policy = MemoryPlanner::select_policy(dgpu, 30ULL * 1024 * 1024 * 1024);
    // 30 GB exceeds the placeholder 8 GB dGPU limit -> fallback to host.
    EXPECT_EQ(policy.memory_type, MemoryType::Host);
    EXPECT_TRUE(policy.allow_offload);
}
