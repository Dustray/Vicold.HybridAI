#include "core/device.h"
#include "core/device_manager.h"
#include "core/dtype.h"
#include "core/shape.h"
#include "core/tensor.h"

#include <gtest/gtest.h>

using namespace hybridai;

TEST(CoreTest, ShapeNumel) {
    Shape shape{2, 3, 4};
    EXPECT_EQ(shape.ndim(), 3);
    EXPECT_EQ(shape.numel(), 24);
}

TEST(CoreTest, DTypeSize) {
    EXPECT_EQ(SizeOfDType(DType::FP32), 4);
    EXPECT_EQ(SizeOfDType(DType::FP16), 2);
    EXPECT_EQ(SizeOfDType(DType::FP8_E4M3), 1);
}

TEST(CoreTest, DeviceCpu) {
    Device cpu = Device::Cpu();
    EXPECT_TRUE(cpu.is_cpu());
    EXPECT_FALSE(cpu.is_gpu());
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
