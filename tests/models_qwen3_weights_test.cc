#include "models/qwen3_weights.h"

#include <gtest/gtest.h>

#include <filesystem>

namespace hybridai::models {
namespace {

TEST(Qwen3WeightLoaderTest, VisionIsDisabledByDefault) {
    Qwen3WeightLoader loader;
    EXPECT_FALSE(loader.options().enable_vision);
}

TEST(Qwen3WeightLoaderTest, RejectsVisionDuringTextOnlyPhase) {
    Qwen3WeightLoader loader;
    Qwen3LoadOptions options;
    options.enable_vision = true;
    const Status status = loader.open("unused", Qwen3Config{}, options);
    EXPECT_EQ(status.code(), StatusCode::NotImplemented);
    EXPECT_FALSE(loader.is_open());
}

TEST(Qwen3WeightLoaderTest, LoadsRealSharedWeightsWhenAvailable) {
    const std::filesystem::path path = "E:/models/Qwen3.8-27B-FP8";
    if (!std::filesystem::exists(path / "config.json")) {
        GTEST_SKIP() << "Downloaded Qwen3.8 model is not available";
    }

    Qwen3WeightLoader loader;
    ASSERT_TRUE(loader.open(path.string()).ok());
    EXPECT_FALSE(loader.options().enable_vision);

    Qwen3SharedWeights weights;
    ASSERT_TRUE(loader.load_shared(Device::Cpu(), &weights).ok());
    EXPECT_EQ(weights.embed_tokens.dtype(), DType::BF16);
    EXPECT_EQ(weights.embed_tokens.shape(), Shape({248320, 5120}));
    EXPECT_EQ(weights.final_norm.shape(), Shape({5120}));
    EXPECT_EQ(weights.lm_head.shape(), Shape({248320, 5120}));
}

TEST(Qwen3WeightLoaderTest, LoadsRealDeltaNetLayerWhenAvailable) {
    const std::filesystem::path path = "E:/models/Qwen3.8-27B-FP8";
    if (!std::filesystem::exists(path / "config.json")) {
        GTEST_SKIP() << "Downloaded Qwen3.8 model is not available";
    }

    Qwen3WeightLoader loader;
    ASSERT_TRUE(loader.open(path.string()).ok());

    Qwen3LayerWeights weights;
    const Status layer_status = loader.load_layer(0, Device::Cpu(), &weights);
    ASSERT_TRUE(layer_status.ok()) << layer_status.message();
    EXPECT_FALSE(weights.is_attention_layer);
    EXPECT_EQ(weights.input_layernorm.shape(), Shape({5120}));
    EXPECT_EQ(weights.in_proj_qkv.values.shape(), Shape({10240, 5120}));
    EXPECT_EQ(weights.in_proj_qkv.scales.shape(), Shape({80, 40}));
    EXPECT_EQ(weights.mlp_gate_proj.values.shape(), Shape({17408, 5120}));
    EXPECT_TRUE(weights.validate().ok());
}

TEST(Qwen3WeightLoaderTest, LoadsRealAttentionLayerWhenAvailable) {
    const std::filesystem::path path = "E:/models/Qwen3.8-27B-FP8";
    if (!std::filesystem::exists(path / "config.json")) {
        GTEST_SKIP() << "Downloaded Qwen3.8 model is not available";
    }

    Qwen3WeightLoader loader;
    ASSERT_TRUE(loader.open(path.string()).ok());

    Qwen3LayerWeights weights;
    const Status layer_status = loader.load_layer(3, Device::Cpu(), &weights);
    ASSERT_TRUE(layer_status.ok()) << layer_status.message();
    EXPECT_TRUE(weights.is_attention_layer);
    EXPECT_EQ(weights.q_proj.values.dtype(), DType::FP8_E4M3);
    EXPECT_EQ(weights.q_proj.values.shape(), Shape({12288, 5120}));
    EXPECT_EQ(weights.k_proj.values.shape(), Shape({1024, 5120}));
    EXPECT_EQ(weights.v_proj.values.shape(), Shape({1024, 5120}));
    EXPECT_TRUE(weights.mlp_down_proj.validate().ok());
}

} // namespace
} // namespace hybridai::models