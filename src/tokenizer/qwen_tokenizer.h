#pragma once

#include "core/status.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace hybridai::tokenizer {

// Minimal ByteLevel BPE tokenizer for Qwen3 models.
// Loads tokenizer.json produced by Hugging Face tokenizers and provides
// text -> ids and ids -> text conversions needed by the inference demo.
class QwenTokenizer {
public:
    QwenTokenizer() = default;

    // Load tokenizer.json from a model directory.
    Status load(const std::string& model_dir);

    bool is_loaded() const noexcept { return !vocab_.empty(); }

    // Encode text to token ids. Special tokens present in the text are
    // recognized when add_special_tokens is true.
    std::vector<int64_t> encode(const std::string& text,
                                bool add_special_tokens = true) const;

    // Decode token ids back to text. Added tokens are decoded verbatim.
    std::string decode(const std::vector<int64_t>& ids,
                       bool skip_special_tokens = false) const;

    // Helpers for applying the Qwen3 chat template without pulling in a
    // full jinja engine.
    std::string build_chat_prompt(
        const std::vector<std::pair<std::string, std::string>>& messages,
        bool add_generation_prompt = true,
        bool enable_thinking = true) const;

    int64_t eos_token_id() const noexcept { return eos_token_id_; }

private:
    // Build the canonical GPT-2 ByteLevel byte <-> unicode mapping.
    static std::unordered_map<uint8_t, char32_t> build_bytes_to_unicode();
    static std::unordered_map<char32_t, uint8_t> build_unicode_to_bytes(
        const std::unordered_map<uint8_t, char32_t>& b2u);

    // Apply ByteLevel encoding to a UTF-8 string (used before BPE).
    std::string byte_encode(const std::string& text) const;

    // Split text into words using the Qwen3 pre-tokenizer regex logic.
    std::vector<std::string> pre_tokenize(const std::string& text) const;

    // BPE encode a single pre-tokenized word that has already been byte-level
    // encoded.
    std::vector<std::string> bpe(const std::string& word) const;

    // Replace added/special tokens in text with placeholders, returning the
    // segments and their ids.
    struct Segment {
        std::string text;
        bool is_special = false;
        int64_t id = -1;
    };
    std::vector<Segment> split_special_tokens(const std::string& text) const;

    struct PairHash {
        size_t operator()(
            const std::pair<std::string, std::string>& p) const noexcept {
            size_t h1 = std::hash<std::string>{}(p.first);
            size_t h2 = std::hash<std::string>{}(p.second);
            return h1 ^ (h2 + 0x9e3779b97f4a7c15ULL + (h1 << 6) +
                         (h1 >> 2));
        }
    };

    std::unordered_map<std::string, int64_t> vocab_;
    std::vector<std::string> id_to_token_;
    std::unordered_map<std::pair<std::string, std::string>, int64_t,
                       PairHash>
        merges_;
    std::unordered_map<std::string, int64_t> added_tokens_;
    std::vector<bool> added_token_flags_;

    std::unordered_map<uint8_t, char32_t> bytes_to_unicode_;
    std::unordered_map<char32_t, uint8_t> unicode_to_bytes_;

    int64_t eos_token_id_ = -1;
    int64_t pad_token_id_ = -1;
    int64_t bos_token_id_ = -1;
};

} // namespace hybridai::tokenizer
