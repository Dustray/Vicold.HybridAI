#include "tokenizer/qwen_tokenizer.h"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <filesystem>
#include <fstream>
#include <string>

namespace hybridai::tokenizer {
namespace {

class QwenTokenizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        static int sequence = 0;
        directory_ = std::filesystem::temp_directory_path() /
                     ("hybridai_qwen_tokenizer_test_" +
                      std::to_string(sequence++));
        std::filesystem::create_directories(directory_);
    }

    void TearDown() override { std::filesystem::remove_all(directory_); }

    void write_tokenizer(const nlohmann::json& added_tokens) const {
        const nlohmann::json tokenizer = {
            {"model", {{"type", "BPE"},
                       {"vocab", {{"a", 0}, {"b", 1}}},
                       {"merges", nlohmann::json::array()}}},
            {"added_tokens", added_tokens}};
        std::ofstream(directory_ / "tokenizer.json") << tokenizer.dump();
    }

    void write_generation(const nlohmann::json& eos) const {
        const nlohmann::json generation = {
            {"bos_token_id", 100},
            {"eos_token_id", eos},
            {"pad_token_id", 101}};
        std::ofstream(directory_ / "generation_config.json")
            << generation.dump();
    }

    std::filesystem::path directory_;
};

TEST_F(QwenTokenizerTest, AddedTokensAboveBaseVocabularyDecodeVerbatim) {
    write_tokenizer(nlohmann::json::array({
        {{"id", 100}, {"content", "<|im_end|>"}, {"special", true}},
        {{"id", 102}, {"content", "<think>"}, {"special", false}}}));
    write_generation(nlohmann::json::array({100, 103}));

    QwenTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(directory_.string()).ok());
    EXPECT_EQ(tokenizer.decode({0, 100, 102, 1}, false),
              "a<|im_end|><think>b");
    EXPECT_EQ(tokenizer.decode({0, 100, 102, 1}, true), "a<think>b");
}

TEST_F(QwenTokenizerTest, LoadsScalarAndArrayGenerationTokenIds) {
    write_tokenizer(nlohmann::json::array());
    write_generation(nlohmann::json::array({100, 103}));

    QwenTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(directory_.string()).ok());
    EXPECT_EQ(tokenizer.eos_token_ids(), (std::vector<int64_t>{100, 103}));
    EXPECT_TRUE(tokenizer.is_eos_token(100));
    EXPECT_TRUE(tokenizer.is_eos_token(103));
    EXPECT_FALSE(tokenizer.is_eos_token(101));
    EXPECT_EQ(tokenizer.bos_token_id(), 100);
    EXPECT_EQ(tokenizer.pad_token_id(), 101);

    write_generation(102);
    ASSERT_TRUE(tokenizer.load(directory_.string()).ok());
    EXPECT_EQ(tokenizer.eos_token_ids(), (std::vector<int64_t>{102}));
}

TEST_F(QwenTokenizerTest, ReloadClearsAddedTokenState) {
    write_tokenizer(nlohmann::json::array({
        {{"id", 100}, {"content", "<old>"}, {"special", false}}}));
    QwenTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(directory_.string()).ok());
    EXPECT_EQ(tokenizer.decode({100}, false), "<old>");

    write_tokenizer(nlohmann::json::array({
        {{"id", 101}, {"content", "<new>"}, {"special", false}}}));
    ASSERT_TRUE(tokenizer.load(directory_.string()).ok());
    EXPECT_TRUE(tokenizer.decode({100}, false).empty());
    EXPECT_EQ(tokenizer.decode({101}, false), "<new>");
}

TEST_F(QwenTokenizerTest, NonSpecialAddedTokenEncodesAtomically) {
    write_tokenizer(nlohmann::json::array({
        {{"id", 102}, {"content", "<think>"}, {"special", false}}}));
    QwenTokenizer tokenizer;
    ASSERT_TRUE(tokenizer.load(directory_.string()).ok());
    EXPECT_EQ(tokenizer.encode("a<think>b", false),
              (std::vector<int64_t>{0, 102, 1}));
}

} // namespace
} // namespace hybridai::tokenizer