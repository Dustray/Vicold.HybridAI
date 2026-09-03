#include "backends/backend_registry.h"
#include "backends/interface/backend.h"
#include "ops/registry.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

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
    EXPECT_TRUE(backend->is_available());
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

TEST(BackendBufferTest, CpuCopyToOffsetCopiesSelectedRange) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    const std::vector<uint8_t> source = {10, 20, 30, 40, 50, 60};
    auto source_buffer = backend->create_buffer(source.size(), MemoryType::Host);
    auto destination_buffer = backend->create_buffer(8, MemoryType::Host);
    ASSERT_NE(source_buffer, nullptr);
    ASSERT_NE(destination_buffer, nullptr);
    ASSERT_TRUE(backend->memcpy_h2d(source_buffer.get(), source.data(),
                                    source.size()).ok());
    ASSERT_TRUE(backend->memset(destination_buffer.get(), 0, 8).ok());

    ASSERT_TRUE(backend->copy_to_offset(destination_buffer.get(), 2,
                                        source_buffer.get(), 1, 3).ok());
    const auto* actual = static_cast<const uint8_t*>(destination_buffer->data());
    ASSERT_NE(actual, nullptr);
    const std::vector<uint8_t> expected = {0, 0, 20, 30, 40, 0, 0, 0};
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_EQ(actual[index], expected[index]) << "index=" << index;
    }
}

TEST(BackendComputeTest, HipFp32GemmMatchesReference) {
    InitializeBuiltinBackends();
    Device hip_device(0, DeviceType::DiscreteGPU, "hip", false);
    auto backend = BackendRegistry::instance().create_backend(hip_device);
    ASSERT_NE(backend, nullptr);
    if (!backend->is_available()) {
        GTEST_SKIP() << "No usable HIP device is available";
    }

    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t k = 4;
    const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> b = {1, 0, 2, -1, 2, 1, 0, 3, -2, 1, 1, 2};
    std::vector<float> expected(m * n, 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t column = 0; column < n; ++column) {
            for (int64_t inner = 0; inner < k; ++inner) {
                expected[row * n + column] +=
                    a[row * k + inner] * b[inner * n + column];
            }
        }
    }

    auto a_buffer = backend->create_buffer(a.size() * sizeof(float),
                                            MemoryType::Device);
    auto b_buffer = backend->create_buffer(b.size() * sizeof(float),
                                            MemoryType::Device);
    auto c_buffer = backend->create_buffer(expected.size() * sizeof(float),
                                            MemoryType::Device);
    ASSERT_NE(a_buffer, nullptr);
    ASSERT_NE(b_buffer, nullptr);
    ASSERT_NE(c_buffer, nullptr);
    ASSERT_TRUE(backend->memcpy_h2d(a_buffer.get(), a.data(),
                                     a.size() * sizeof(float)).ok());
    ASSERT_TRUE(backend->memcpy_h2d(b_buffer.get(), b.data(),
                                     b.size() * sizeof(float)).ok());

    ASSERT_TRUE(backend->gemm(c_buffer.get(), a_buffer.get(), b_buffer.get(),
                              DType::FP32, DType::FP32, DType::FP32,
                              DType::FP32,
                              false, false, m, n, k, 1.0f, 0.0f)
                    .ok());
    ASSERT_TRUE(backend->synchronize().ok());

    std::vector<float> actual(expected.size());
    ASSERT_TRUE(backend->memcpy_d2h(actual.data(), c_buffer.get(),
                                     actual.size() * sizeof(float))
                    .ok());
    ASSERT_TRUE(backend->synchronize().ok());
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-4f);
    }
}

TEST(BackendComputeTest, CpuTransposeWeightGemmMatchesReference) {
    InitializeBuiltinBackends();
    auto backend = BackendRegistry::instance().create_backend(Device::Cpu());
    ASSERT_NE(backend, nullptr);

    // A is [M,K]. B is stored as [N,K], and trans_b=true must compute A*B^T.
    constexpr int64_t m = 2;
    constexpr int64_t n = 3;
    constexpr int64_t k = 4;
    const std::vector<float> a = {1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<float> b = {
        1, 0, 2, -1,  // output row 0
        2, 1, 0, 3,   // output row 1
        -2, 1, 1, 2,  // output row 2
    };
    std::vector<float> expected(m * n, 0.0f);
    for (int64_t row = 0; row < m; ++row) {
        for (int64_t column = 0; column < n; ++column) {
            for (int64_t inner = 0; inner < k; ++inner) {
                expected[row * n + column] +=
                    a[row * k + inner] * b[column * k + inner];
            }
        }
    }

    auto a_buffer = backend->create_buffer(a.size() * sizeof(float),
                                            MemoryType::Host);
    auto b_buffer = backend->create_buffer(b.size() * sizeof(float),
                                            MemoryType::Host);
    auto c_buffer = backend->create_buffer(expected.size() * sizeof(float),
                                            MemoryType::Host);
    ASSERT_NE(a_buffer, nullptr);
    ASSERT_NE(b_buffer, nullptr);
    ASSERT_NE(c_buffer, nullptr);
    ASSERT_TRUE(backend->memcpy_h2d(a_buffer.get(), a.data(),
                                     a.size() * sizeof(float)).ok());
    ASSERT_TRUE(backend->memcpy_h2d(b_buffer.get(), b.data(),
                                     b.size() * sizeof(float)).ok());
    ASSERT_TRUE(backend->gemm(c_buffer.get(), a_buffer.get(), b_buffer.get(),
                              DType::FP32, DType::FP32, DType::FP32,
                              DType::FP32, false, true, m, n, k, 1.0f, 0.0f)
                    .ok());

    std::vector<float> actual(expected.size());
    ASSERT_TRUE(backend->memcpy_d2h(actual.data(), c_buffer.get(),
                                     actual.size() * sizeof(float))
                    .ok());
    for (size_t index = 0; index < expected.size(); ++index) {
        EXPECT_NEAR(actual[index], expected[index], 1e-4f);
    }
}

} // namespace
} // namespace hybridai
