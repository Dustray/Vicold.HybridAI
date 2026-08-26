#include "backends/backend_registry.h"
#include "core/device.h"
#include "core/device_manager.h"
#include "core/dtype.h"
#include "core/platform.h"
#include "core/shape.h"
#include "core/tensor.h"

#include <gtest/gtest.h>

#include <cstring>
#include <vector>

using namespace hybridai;

TEST(CoreTest, ShapeNumel) {
    Shape shape{2, 3, 4};
    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape.numel(), 24);
}

TEST(CoreTest, DTypeSize) {
    EXPECT_EQ(SizeOfDType(DType::FP32), 4);
    EXPECT_EQ(SizeOfDType(DType::FP16), 2);
    EXPECT_EQ(SizeOfDType(DType::BF16), 2);
    EXPECT_EQ(SizeOfDType(DType::FP8_E4M3), 1);
    EXPECT_EQ(SizeOfDType(DType::INT8), 1);
    EXPECT_EQ(SizeOfDType(DType::INT32), 4);
}

TEST(CoreTest, DeviceCpu) {
    Device cpu = Device::Cpu();
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_gpu());
    EXPECT_EQ(cpu.backend(), "cpu");
}

TEST(CoreTest, DeviceManagerInit) {
    DeviceManager::instance().initialize();
    EXPECT_GE(DeviceManager::instance().devices().size(), 1);
}

TEST(CoreTest, TensorEmpty) {
    Tensor t;
    EXPECT_EQ(t.numel(), 0);
    EXPECT_EQ(t.data(), nullptr);
}

TEST(CoreTest, TensorCpuCreateAndRead) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    ASSERT_NE(backend, nullptr);

    Shape shape{2, 3};
    auto buffer = backend->create_buffer(shape.numel() * sizeof(float),
                                       MemoryType::Host);
    ASSERT_NE(buffer, nullptr);

    float* data = static_cast<float*>(buffer->data());
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i);

    Tensor t(shape, DType::FP32, cpu, buffer);
    EXPECT_EQ(t.numel(), 6);
    EXPECT_TRUE(t.is_contiguous());
    EXPECT_EQ(std::memcmp(t.data(), data, t.nbytes()), 0);
}

TEST(CoreTest, TensorToSameDevice) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    ASSERT_NE(backend, nullptr);

    Shape shape{2, 3};
    auto buffer = backend->create_buffer(shape.numel() * sizeof(float),
                                         MemoryType::Host);
    float* data = static_cast<float*>(buffer->data());
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i);

    Tensor t(shape, DType::FP32, cpu, buffer);
    Tensor same = t.to(cpu);
    EXPECT_EQ(same.data(), t.data());
}

TEST(CoreTest, TensorToUnsupportedDevice) {
    Device cpu = Device::Cpu();
    Device fake(0, DeviceType::DiscreteGPU, "nonexistent_backend", false);
    auto backend = BackendRegistry::instance().create_backend(cpu);

    Shape shape{2, 3};
    auto buffer = backend->create_buffer(shape.numel() * sizeof(float),
                                       MemoryType::Host);
    float* data = static_cast<float*>(buffer->data());
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i);

    Tensor t(shape, DType::FP32, cpu, buffer);
    Tensor moved = t.to(fake);
    EXPECT_EQ(moved.data(), nullptr);
}

TEST(CoreTest, TensorContiguousCopy) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);

    // Create a non-contiguous tensor: shape (2,3) with stride (3,1) is
    // contiguous; to get non-contiguous we use a slice-like view by keeping
    // a larger backing buffer and a smaller logical shape with larger strides.
    // Simulate shape (2,2) inside a (2,3) matrix with stride (3,1).
    Shape shape{2, 3};
    auto buffer = backend->create_buffer(shape.numel() * sizeof(float),
                                         MemoryType::Host);
    float* data = static_cast<float*>(buffer->data());
    // row 0: 0 1 2 ; row 1: 3 4 5
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i);

    // Create a transposed view: logical shape (3,2), strides (1,3)
    Shape view_shape{3, 2};
    std::vector<int64_t> strides{1, 3};
    Tensor view(view_shape, DType::FP32, cpu, buffer, strides);
    ASSERT_FALSE(view.is_contiguous());

    Tensor cont = view.contiguous();
    ASSERT_TRUE(cont.is_contiguous());
    ASSERT_NE(cont.data(), nullptr);

    const float* result = static_cast<const float*>(cont.data());
    // Transpose of [[0,1,2],[3,4,5]] is [[0,3],[1,4],[2,5]]
    EXPECT_FLOAT_EQ(result[0], 0.0f);
    EXPECT_FLOAT_EQ(result[1], 3.0f);
    EXPECT_FLOAT_EQ(result[2], 1.0f);
    EXPECT_FLOAT_EQ(result[3], 4.0f);
    EXPECT_FLOAT_EQ(result[4], 2.0f);
    EXPECT_FLOAT_EQ(result[5], 5.0f);
}

TEST(CoreTest, TensorContiguousAlready) {
    Device cpu = Device::Cpu();
    auto backend = BackendRegistry::instance().create_backend(cpu);
    Shape shape{2, 3};
    auto buffer = backend->create_buffer(shape.numel() * sizeof(float),
                                       MemoryType::Host);
    Tensor t(shape, DType::FP32, cpu, buffer);
    Tensor cont = t.contiguous();
    EXPECT_EQ(cont.data(), t.data());
}

TEST(CoreTest, PlatformPathsAndTime) {
    auto t0 = platform::monotonic_us();
    EXPECT_GT(platform::cpu_count(), 0);
    auto t1 = platform::monotonic_us();
    EXPECT_GE(t1, t0);

    std::string exe_dir = platform::executable_dir();
    EXPECT_FALSE(exe_dir.empty());
    EXPECT_TRUE(platform::file_exists(exe_dir));
}

TEST(CoreTest, PlatformFileSize) {
    std::string exe_dir = platform::executable_dir();
    EXPECT_FALSE(exe_dir.empty());
}
