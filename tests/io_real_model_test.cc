#include "io/safetensor_loader.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace hybridai::io {
namespace {

TEST(SafeTensorLoaderTest, ReadsDownloadedQwen38FirstShardWhenAvailable) {
    const std::filesystem::path path =
        "E:/models/Qwen3.8-27B-FP8/layers-0.safetensors";
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "Downloaded Qwen3.8 model is not available";
    }

    SafeTensorLoader loader;
    ASSERT_TRUE(loader.open(path.string()).ok());
    const auto* weight = loader.tensor_info(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    ASSERT_NE(weight, nullptr);
    EXPECT_EQ(weight->dtype, DType::FP8_E4M3);
    EXPECT_EQ(weight->shape, Shape({10240, 5120}));

    const auto* scale = loader.tensor_info(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv");
    ASSERT_NE(scale, nullptr);
    EXPECT_EQ(scale->dtype, DType::BF16);
    EXPECT_EQ(scale->shape, Shape({80, 40}));
}

} // namespace
} // namespace hybridai::io