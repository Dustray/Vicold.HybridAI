#include "models/qwen3_config.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <vector>

namespace hybridai::models {
namespace {

TEST(Qwen3ConfigTest, LoadsTextConfigAndRopeParameters) {
    const auto path = std::filesystem::temp_directory_path() /
                      "hybridai_qwen3_config_test.json";
    nlohmann::json config = {
        {"text_config", {
            {"hidden_size", 5120}, {"num_hidden_layers", 64},
            {"num_attention_heads", 24}, {"num_key_value_heads", 4},
            {"head_dim", 256}, {"intermediate_size", 17408},
            {"vocab_size", 248320}, {"max_position_embeddings", 262144},
            {"rope_parameters", {{"rope_theta", 10000000.0},
                                  {"partial_rotary_factor", 0.25}}}
        }}
    };
    std::ofstream(path) << config.dump();

    Qwen3Config parsed;
    ASSERT_TRUE(parsed.load_json(path.string()).ok());
    EXPECT_EQ(parsed.hidden_size, 5120);
    EXPECT_EQ(parsed.num_hidden_layers, 64);
    EXPECT_EQ(parsed.num_key_value_heads, 4);
    EXPECT_EQ(parsed.rope_head_dim, 64);
    EXPECT_FLOAT_EQ(parsed.rope_theta, 10000000.0f);
    ASSERT_EQ(parsed.layer_types.size(), 64u);
    EXPECT_EQ(parsed.layer_types[0], Qwen3LayerType::LinearAttention);
    EXPECT_EQ(parsed.layer_types[3], Qwen3LayerType::FullAttention);
    std::filesystem::remove(path);
}

TEST(Qwen3ConfigTest, LoadsExplicitScheduleBehaviorAndTokenMetadata) {
    const auto path = std::filesystem::temp_directory_path() /
                      "hybridai_qwen3_schedule_test.json";
    std::vector<std::string> layer_types;
    for (int index = 0; index < 8; ++index) {
        layer_types.push_back((index + 1) % 4 == 0
                                  ? "full_attention"
                                  : "linear_attention");
    }
    nlohmann::json config = {
        {"tie_word_embeddings", true},
        {"eos_token_id", nlohmann::json::array({8, 9})},
        {"text_config", {
            {"num_hidden_layers", 8},
            {"layer_types", layer_types},
            {"full_attention_interval", 4},
            {"attn_output_gate", true},
            {"output_gate_type", "swish"},
            {"mamba_ssm_dtype", "float32"},
            {"use_cache", false},
            {"bos_token_id", 7},
            {"pad_token_id", nullptr}
        }}
    };
    std::ofstream(path) << config.dump();

    Qwen3Config parsed;
    ASSERT_TRUE(parsed.load_json(path.string()).ok());
    EXPECT_EQ(parsed.layer_types.size(), 8u);
    EXPECT_EQ(parsed.layer_types[2], Qwen3LayerType::LinearAttention);
    EXPECT_EQ(parsed.layer_types[3], Qwen3LayerType::FullAttention);
    EXPECT_FALSE(parsed.use_cache);
    EXPECT_TRUE(parsed.tie_word_embeddings);
    EXPECT_EQ(parsed.bos_token_id, 7);
    EXPECT_EQ(parsed.pad_token_id, -1);
    EXPECT_EQ(parsed.eos_token_ids, (std::vector<int64_t>{8, 9}));
    std::filesystem::remove(path);
}

TEST(Qwen3ConfigTest, RejectsUnknownOrConflictingLayerSchedule) {
    const auto path = std::filesystem::temp_directory_path() /
                      "hybridai_qwen3_bad_schedule_test.json";
    nlohmann::json config = {
        {"num_hidden_layers", 4},
        {"full_attention_interval", 4},
        {"layer_types", {"linear_attention", "linear_attention",
                         "linear_attention", "unknown_attention"}}
    };
    std::ofstream(path) << config.dump();
    Qwen3Config parsed;
    EXPECT_EQ(parsed.load_json(path.string()).code(), StatusCode::InvalidModel);

    config["layer_types"] = {"linear_attention", "linear_attention",
                             "linear_attention", "linear_attention"};
    std::ofstream(path) << config.dump();
    EXPECT_EQ(parsed.load_json(path.string()).code(), StatusCode::InvalidModel);
    std::filesystem::remove(path);
}

TEST(Qwen3ConfigTest, RejectsUnsupportedSsmDtype) {
    const auto path = std::filesystem::temp_directory_path() /
                      "hybridai_qwen3_ssm_dtype_test.json";
    nlohmann::json config = {{"num_hidden_layers", 4},
                             {"full_attention_interval", 4},
                             {"mamba_ssm_dtype", "bfloat16"}};
    std::ofstream(path) << config.dump();
    Qwen3Config parsed;
    EXPECT_EQ(parsed.load_json(path.string()).code(), StatusCode::InvalidModel);
    std::filesystem::remove(path);
}

TEST(Qwen3ConfigTest, LoadsDownloadedQwen38ConfigWhenAvailable) {
    const std::filesystem::path path =
        "/public/home/panyq/yiny/modelscope/models/Qwen--Qwen3.8-27B/snapshots/master/config.json";
    if (!std::filesystem::exists(path)) {
        GTEST_SKIP() << "Downloaded Qwen3.8 model is not available";
    }

    Qwen3Config parsed;
    ASSERT_TRUE(parsed.load_json(path.string()).ok());
    EXPECT_EQ(parsed.hidden_size, 5120);
    EXPECT_EQ(parsed.num_hidden_layers, 64);
    EXPECT_EQ(parsed.num_attention_heads, 24);
    EXPECT_EQ(parsed.num_key_value_heads, 4);
    EXPECT_EQ(parsed.head_dim, 256);
    EXPECT_EQ(parsed.rope_head_dim, 64);
    EXPECT_EQ(parsed.vocab_size, 248320);
    ASSERT_EQ(parsed.layer_types.size(), 64u);
    EXPECT_EQ(parsed.layer_types[0], Qwen3LayerType::LinearAttention);
    EXPECT_EQ(parsed.layer_types[3], Qwen3LayerType::FullAttention);
    EXPECT_EQ(parsed.full_attention_interval, 4);
    EXPECT_TRUE(parsed.attn_output_gate);
    EXPECT_EQ(parsed.output_gate_type, "swish");
    EXPECT_EQ(parsed.mamba_ssm_dtype, "float32");
    EXPECT_TRUE(parsed.use_cache);
}

} // namespace
} // namespace hybridai::models