#include "models/qwen3_config.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>

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
    std::filesystem::remove(path);
}

TEST(Qwen3ConfigTest, LoadsDownloadedQwen38ConfigWhenAvailable) {
    const std::filesystem::path path =
        "E:/models/Qwen3.8-27B-FP8/config.json";
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
}

} // namespace
} // namespace hybridai::models