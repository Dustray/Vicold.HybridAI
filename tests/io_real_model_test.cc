#include "io/safetensor_loader.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <unordered_map>

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

TEST(SafeTensorLoaderTest, OpensDownloadedQwen38ShardedModelWhenAvailable) {
    const std::filesystem::path path = "E:/models/Qwen3.8-27B-FP8";
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        GTEST_SKIP() << "Downloaded sharded Qwen3.8 model is not available";
    }

    SafeTensorLoader loader;
    ASSERT_TRUE(loader.open_model(path.string()).ok());
    const auto names = loader.tensor_names();
    EXPECT_GT(names.size(), 1000u);

    const auto* first_norm = loader.tensor_info(
        "model.language_model.layers.0.input_layernorm.weight");
    ASSERT_NE(first_norm, nullptr);
    EXPECT_EQ(first_norm->dtype, DType::BF16);
    EXPECT_EQ(first_norm->shape, Shape({5120}));

    const auto* last_weight = loader.tensor_info(
        "model.language_model.layers.63.mlp.up_proj.weight");
    ASSERT_NE(last_weight, nullptr);
    EXPECT_EQ(last_weight->dtype, DType::FP8_E4M3);
    EXPECT_EQ(last_weight->shape, Shape({17408, 5120}));
}

TEST(SafeTensorLoaderTest, ValidatesDownloadedQwen38ScaleShapesWhenAvailable) {
    const std::filesystem::path path = "E:/models/Qwen3.8-27B-FP8";
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        GTEST_SKIP() << "Downloaded sharded Qwen3.8 model is not available";
    }

    SafeTensorLoader loader;
    ASSERT_TRUE(loader.open_model(path.string()).ok());
    const auto* weight = loader.tensor_info(
        "model.language_model.layers.0.mlp.gate_proj.weight");
    const auto* scale = loader.tensor_info(
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv");
    ASSERT_NE(weight, nullptr);
    ASSERT_NE(scale, nullptr);
    ASSERT_EQ(weight->dtype, DType::FP8_E4M3);
    ASSERT_EQ(scale->dtype, DType::BF16);
    EXPECT_EQ(scale->shape,
              Shape({(17408 + 127) / 128, (5120 + 127) / 128}));
}

} // namespace
} // namespace hybridai::io