#include "io/safetensor_loader.h"

#include <gtest/gtest.h>

#include <cstdlib>
#include <filesystem>
#include <unordered_map>

namespace hybridai::io {
namespace {

std::filesystem::path real_qwen_model_path() {
    if (const char* configured = std::getenv("HYBRIDAI_QWEN_MODEL_DIR");
        configured != nullptr && configured[0] != '\0') {
        return configured;
    }
    const std::filesystem::path linux_path =
        "/public/home/panyq/yiny/modelscope/models/"
        "Qwen--Qwen3.8-27B/snapshots/master";
    if (std::filesystem::exists(linux_path / "config.json")) {
        return linux_path;
    }
    return "E:/models/Qwen3.8-27B-FP8";
}

void expect_dense_or_fp8_weight(const SafeTensorInfo& weight) {
    EXPECT_TRUE(weight.dtype == DType::BF16 ||
                weight.dtype == DType::FP8_E4M3);
}

TEST(SafeTensorLoaderTest, ReadsDownloadedQwen38FirstShardWhenAvailable) {
    const std::filesystem::path path = real_qwen_model_path();
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        GTEST_SKIP() << "Qwen model is not available at " << path;
    }

    SafeTensorLoader loader;
    ASSERT_TRUE(loader.open_model(path.string()).ok());
    const auto* weight = loader.tensor_info(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight");
    ASSERT_NE(weight, nullptr);
    expect_dense_or_fp8_weight(*weight);
    EXPECT_EQ(weight->shape, Shape({10240, 5120}));

    const auto* scale = loader.tensor_info(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight_scale_inv");
    if (weight->dtype == DType::FP8_E4M3) {
        ASSERT_NE(scale, nullptr);
        EXPECT_EQ(scale->dtype, DType::BF16);
        EXPECT_EQ(scale->shape, Shape({80, 40}));
    } else {
        EXPECT_EQ(scale, nullptr);
    }
}

TEST(SafeTensorLoaderTest, OpensDownloadedQwen38ShardedModelWhenAvailable) {
    const std::filesystem::path path = real_qwen_model_path();
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        GTEST_SKIP() << "Sharded Qwen model is not available at " << path;
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
    expect_dense_or_fp8_weight(*last_weight);
    EXPECT_EQ(last_weight->shape, Shape({17408, 5120}));
}

TEST(SafeTensorLoaderTest, ValidatesDownloadedQwen38ScaleShapesWhenAvailable) {
    const std::filesystem::path path = real_qwen_model_path();
    if (!std::filesystem::exists(path / "model.safetensors.index.json")) {
        GTEST_SKIP() << "Sharded Qwen model is not available at " << path;
    }

    SafeTensorLoader loader;
    ASSERT_TRUE(loader.open_model(path.string()).ok());
    const auto* weight = loader.tensor_info(
        "model.language_model.layers.0.mlp.gate_proj.weight");
    const auto* scale = loader.tensor_info(
        "model.language_model.layers.0.mlp.gate_proj.weight_scale_inv");
    ASSERT_NE(weight, nullptr);
    expect_dense_or_fp8_weight(*weight);
    if (weight->dtype == DType::FP8_E4M3) {
        ASSERT_NE(scale, nullptr);
        ASSERT_EQ(scale->dtype, DType::BF16);
        EXPECT_EQ(scale->shape,
                  Shape({(17408 + 127) / 128, (5120 + 127) / 128}));
    } else {
        EXPECT_EQ(scale, nullptr);
    }
}

} // namespace
} // namespace hybridai::io