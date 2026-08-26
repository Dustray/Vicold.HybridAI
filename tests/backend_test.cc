#include "backends/backend_registry.h"
#include "backends/interface/backend.h"
#include "ops/registry.h"

#include <gtest/gtest.h>

namespace hybridai {
namespace {

TEST(BackendRegistryTest, CpuBackendRegistered) {
    InitializeBuiltinBackends();
    EXPECT_TRUE(BackendRegistry::instance().has_backend("cpu"));
}

TEST(BackendRegistryTest, HipBackendRegisteredAsStub) {
    InitializeBuiltinBackends();
    EXPECT_TRUE(BackendRegistry::instance().has_backend("hip"));
}

TEST(BackendRegistryTest, CreateCpuBackend) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    EXPECT_STREQ(backend->name(), "cpu");
}

TEST(BackendRegistryTest, HipBackendCanBeCreated) {
    InitializeBuiltinBackends();
    Device hip_device(0, DeviceType::DiscreteGPU, "hip", true);
    auto backend = BackendRegistry::instance().create_backend(hip_device);
    ASSERT_NE(backend, nullptr);
    EXPECT_STREQ(backend->name(), "hip");
    EXPECT_EQ(backend->device(), hip_device);
}

TEST(BackendRegistryTest, UnknownBackendReturnsNull) {
    InitializeBuiltinBackends();
    Device unknown(0, DeviceType::Unknown, "xpu");
    auto backend = BackendRegistry::instance().create_backend(unknown);
    EXPECT_EQ(backend, nullptr);
}

TEST(BackendKernelTest, CpuRegistersNoopKernel) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);
    backend->register_kernels();

    ops::KernelKey key{"noop", DeviceType::CPU, DType::FP32};
    EXPECT_TRUE(ops::KernelRegistry::instance().has_kernel(key));

    auto fn = ops::KernelRegistry::instance().find_kernel(key);
    ASSERT_NE(fn, nullptr);
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    EXPECT_TRUE(fn(inputs, outputs).ok());
}

TEST(BackendKernelTest, HipRegistersNoopKernel) {
    InitializeBuiltinBackends();
    Device hip_device(0, DeviceType::DiscreteGPU, "hip", true);
    auto backend = BackendRegistry::instance().create_backend(hip_device);
    ASSERT_NE(backend, nullptr);
    backend->register_kernels();

    ops::KernelKey key{"noop", DeviceType::DiscreteGPU, DType::FP32};
    EXPECT_TRUE(ops::KernelRegistry::instance().has_kernel(key));

    auto fn = ops::KernelRegistry::instance().find_kernel(key);
    ASSERT_NE(fn, nullptr);
    std::vector<Tensor*> inputs;
    std::vector<Tensor*> outputs;
    EXPECT_TRUE(fn(inputs, outputs).ok());
}

TEST(BackendBufferTest, CpuCreateBuffer) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    auto buffer = backend->create_buffer(64, MemoryType::Host);
    ASSERT_NE(buffer, nullptr);
    EXPECT_NE(buffer->data(), nullptr);
    EXPECT_EQ(buffer->size(), 64u);
    EXPECT_EQ(buffer->memory_type(), MemoryType::Host);

    EXPECT_TRUE(backend->memset(buffer.get(), 0, 64).ok());
    EXPECT_TRUE(backend->synchronize().ok());
}

} // namespace
} // namespace hybridai
