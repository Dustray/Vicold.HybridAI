#include "tokenizer/qwen_tokenizer.h"

#include "core/platform.h"
#include "tokenizer/unicode_utils.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <set>
#include <sstream>
#include <unordered_map>

namespace hybridai::tokenizer {

namespace {

// Convert a UTF-8 string to raw bytes (each char is one byte).
std::vector<uint8_t> utf8_to_bytes(const std::string& s) {
    return std::vector<uint8_t>(s.begin(), s.end());
}

// Convert raw bytes to a UTF-8 string. Bytes are emitted as-is; callers that
// need strict UTF-8 validation should handle invalid sequences themselves.
std::string bytes_to_utf8(const std::vector<uint8_t>& bytes) {
    return std::string(bytes.begin(), bytes.end());
}

} // namespace

std::unordered_map<uint8_t, char32_t>
QwenTokenizer::build_bytes_to_unicode() {
    std::unordered_map<uint8_t, char32_t> b2u;
    std::vector<int> byte_list;
    for (int i = '!'; i <= '~'; ++i) byte_list.push_back(i);
    for (int i = 0xA1; i <= 0xAC; ++i) byte_list.push_back(i);
    for (int i = 0xAE; i <= 0xFF; ++i) byte_list.push_back(i);
    std::set<int> used(byte_list.begin(), byte_list.end());
    for (int b = 0; b < 256; ++b) {
        if (used.find(b) == used.end()) byte_list.push_back(b);
    }
    for (size_t i = 0; i < byte_list.size(); ++i) {
        int b = byte_list[i];
        char32_t cp;
        if (i < 188) {
            cp = static_cast<char32_t>(b);
        } else {
            cp = static_cast<char32_t>(256 + (i - 188));
        }
        b2u[static_cast<uint8_t>(b)] = cp;
    }
    return b2u;
}

std::unordered_map<char32_t, uint8_t> QwenTokenizer::build_unicode_to_bytes(
    const std::unordered_map<uint8_t, char32_t>& b2u) {
    std::unordered_map<char32_t, uint8_t> u2b;
    for (const auto& kv : b2u) {
        u2b[kv.second] = kv.first;
    }
    return u2b;
}

Status QwenTokenizer::load(const std::string& model_dir) {
    std::string path = platform::join_path(model_dir, "tokenizer.json");
    if (!platform::file_exists(path)) {
        return Status(StatusCode::FileNotFound,
                      "tokenizer.json not found in " + model_dir);
    }

    std::ifstream file(path);
    if (!file) {
        return Status(StatusCode::FileNotFound,
                      "Failed to open tokenizer.json");
    }

    nlohmann::json root;
    try {
        file >> root;
    } catch (const nlohmann::json::exception& e) {
        return Status(StatusCode::InvalidModel,
                      std::string("Invalid tokenizer.json: ") + e.what());
    }

    if (!root.contains("model") || !root["model"].is_object()) {
        return Status(StatusCode::InvalidModel,
                      "tokenizer.json missing model section");
    }
    const auto& model = root["model"];
    if (model.value("type", "") != "BPE") {
        return Status(StatusCode::InvalidModel,
                      "Only BPE tokenizers are supported");
    }

    // Vocabulary: token string -> id.
    const auto& vocab_json = model["vocab"];
    vocab_.reserve(vocab_json.size());
    int64_t max_id = 0;
    for (auto it = vocab_json.begin(); it != vocab_json.end(); ++it) {
        std::string token = it.key();
        int64_t id = it.value().get<int64_t>();
        vocab_[token] = id;
        max_id = std::max(max_id, id);
    }
    id_to_token_.resize(static_cast<size_t>(max_id) + 1);
    for (const auto& kv : vocab_) {
        id_to_token_[static_cast<size_t>(kv.second)] = kv.first;
    }

    // Merges: pair -> rank (order in the merges list).
    // HF tokenizer.json stores each merge as a single string "a b".
    const auto& merges_json = model["merges"];
    for (size_t i = 0; i < merges_json.size(); ++i) {
        const std::string merge_str = merges_json[i].get<std::string>();
        const size_t split = merge_str.find(' ');
        if (split == std::string::npos) continue;
        const std::string first = merge_str.substr(0, split);
        const std::string second = merge_str.substr(split + 1);
        merges_[{first, second}] = static_cast<int64_t>(i);
    }

    // Added tokens (specials).
    added_tokens_.clear();
    if (root.contains("added_tokens") && root["added_tokens"].is_array()) {
        for (const auto& tok : root["added_tokens"]) {
            std::string content = tok.value("content", "");
            int64_t id = tok.value("id", -1);
            bool special = tok.value("special", false);
            if (content.empty() || id < 0) continue;
            added_tokens_[content] = id;
            if (id >= static_cast<int64_t>(added_token_flags_.size())) {
                added_token_flags_.resize(static_cast<size_t>(id) + 1, false);
            }
            added_token_flags_[static_cast<size_t>(id)] = special;
        }
    }

    eos_token_id_ = -1;
    pad_token_id_ = -1;
    bos_token_id_ = -1;
    for (const auto& kv : added_tokens_) {
        if (kv.first == "<|endoftext|>") pad_token_id_ = kv.second;
        if (kv.first == "\n") eos_token_id_ = kv.second;
    }

    bytes_to_unicode_ = build_bytes_to_unicode();
    unicode_to_bytes_ = build_unicode_to_bytes(bytes_to_unicode_);

    return Status::OK();
}

std::string QwenTokenizer::byte_encode(const std::string& text) const {
    std::string encoded;
    encoded.reserve(text.size());
    for (uint8_t b : text) {
        auto it = bytes_to_unicode_.find(b);
        if (it == bytes_to_unicode_.end()) {
            // Should never happen: all 256 bytes have a mapping.
            continue;
        }
        encoded.append(codepoints_to_utf8({it->second}));
    }
    return encoded;
}

std::vector<QwenTokenizer::Segment>
QwenTokenizer::split_special_tokens(const std::string& text) const {
    std::vector<Segment> segments;
    size_t i = 0;
    while (i < text.size()) {
        // Longest match first.
        std::string best_token;
        int64_t best_id = -1;
        for (const auto& kv : added_tokens_) {
            const std::string& tok = kv.first;
            if (tok.empty()) continue;
            if (text.compare(i, tok.size(), tok) == 0 &&
                tok.size() > best_token.size()) {
                best_token = tok;
                best_id = kv.second;
            }
        }
        if (!best_token.empty()) {
            segments.push_back({best_token, true, best_id});
            i += best_token.size();
            continue;
        }
        if (!segments.empty() && !segments.back().is_special) {
            segments.back().text.push_back(text[i]);
        } else {
            segments.push_back({std::string(1, text[i]), false, -1});
        }
        ++i;
    }
    return segments;
}

std::vector<std::string> QwenTokenizer::pre_tokenize(
    const std::string& text) const {
    std::vector<std::string> result;
    const auto cpts = utf8_to_codepoints(text);
    const size_t n = cpts.size();
    if (n == 0) return result;

    auto get_flags = [&](size_t pos) -> CodepointFlags {
        if (pos >= n) return CodepointFlags{};
        return codepoint_flags(cpts[pos]);
    };
    auto get_cpt = [&](size_t pos) -> uint32_t {
        if (pos >= n) return 0;
        return cpts[pos];
    };

    size_t pos = 0;
    size_t prev = 0;
    auto add_token = [&](size_t end) {
        if (end > prev) {
            result.push_back(codepoints_to_utf8(
                std::vector<uint32_t>(cpts.begin() + prev, cpts.begin() + end)));
            prev = end;
        }
    };

    while (pos < n) {
        const uint32_t cpt = get_cpt(pos);
        const auto flags = get_flags(pos);

        // (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (cpt == '\'' && pos + 1 < n) {
            uint32_t cn = get_cpt(pos + 1);
            uint32_t cnl = cn;
            if (cn >= 'A' && cn <= 'Z') cnl = cn - 'A' + 'a';
            if (cnl == 's' || cnl == 't' || cnl == 'm' || cnl == 'd') {
                pos += 2;
                add_token(pos);
                continue;
            }
            if (pos + 2 < n) {
                uint32_t cnn = get_cpt(pos + 2);
                uint32_t cnnl = cnn;
                if (cnn >= 'A' && cnn <= 'Z') cnnl = cnn - 'A' + 'a';
                if ((cnl == 'r' && cnnl == 'e') ||
                    (cnl == 'v' && cnnl == 'e') ||
                    (cnl == 'l' && cnnl == 'l')) {
                    pos += 3;
                    add_token(pos);
                    continue;
                }
            }
        }

        // [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
        // The optional leading char may be whitespace (e.g. " how") or a
        // symbol (e.g. ",word"). Consume exactly one such char, then run
        // through all following letters/marks.
        if (cpt != '\r' && cpt != '\n' && !flags.is_number) {
            auto flags_next = get_flags(pos + 1);
            if (flags.is_letter || flags.is_mark || flags_next.is_letter ||
                flags_next.is_mark) {
                // Always advance at least one codepoint: either the leading
                // optional char or the first letter/mark itself.
                ++pos;
                while (pos < n &&
                       (get_flags(pos).is_letter || get_flags(pos).is_mark)) {
                    ++pos;
                }
                add_token(pos);
                continue;
            }
        }

        // \p{N}
        if (flags.is_number) {
            ++pos;
            add_token(pos);
            continue;
        }

        // <space>?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
        auto flags2 = (cpt == ' ' ? get_flags(pos + 1) : flags);
        if (!flags2.is_whitespace && !flags2.is_letter && !flags2.is_mark &&
            !flags2.is_number && flags2.is_defined) {
            if (cpt == ' ') ++pos;
            while (pos < n) {
                auto f = get_flags(pos);
                if (f.is_whitespace || f.is_letter || f.is_mark ||
                    f.is_number || !f.is_defined) {
                    break;
                }
                ++pos;
            }
            while (pos < n && (get_cpt(pos) == '\r' || get_cpt(pos) == '\n')) {
                ++pos;
            }
            add_token(pos);
            continue;
        }

        // Whitespace handling: \s*[\r\n]+ | \s+(?!\S) | \s+
        size_t num_whitespace = 0;
        size_t last_rn_end = 0;
        while (pos + num_whitespace < n &&
               get_flags(pos + num_whitespace).is_whitespace) {
            uint32_t c2 = get_cpt(pos + num_whitespace);
            if (c2 == '\r' || c2 == '\n') {
                last_rn_end = pos + num_whitespace + 1;
            }
            ++num_whitespace;
        }

        // \s*[\r\n]+
        if (last_rn_end > 0) {
            pos = last_rn_end;
            add_token(pos);
            continue;
        }

        // \s+(?!\S): trailing whitespace before a non-space consumes all but
        // the last space.
        if (num_whitespace > 1 && pos + num_whitespace < n) {
            pos += num_whitespace - 1;
            add_token(pos);
            continue;
        }

        // \s+
        if (num_whitespace > 0) {
            pos += num_whitespace;
            add_token(pos);
            continue;
        }

        // No match: consume one codepoint and continue.
        ++pos;
        add_token(pos);
    }

    if (result.empty() && !text.empty()) {
        result.push_back(text);
    }
    return result;
}

std::vector<std::string> QwenTokenizer::bpe(const std::string& word) const {
    // word is already ByteLevel encoded (each character is one original byte).
    std::vector<std::string> word_tokens;
    const auto cps = utf8_to_codepoints(word);
    word_tokens.reserve(cps.size());
    for (uint32_t c : cps) {
        word_tokens.push_back(codepoints_to_utf8({c}));
    }

    if (word_tokens.size() <= 1) return word_tokens;

    auto get_pair_rank = [&](size_t i) -> int64_t {
        auto it = merges_.find({word_tokens[i], word_tokens[i + 1]});
        if (it == merges_.end()) {
            return std::numeric_limits<int64_t>::max();
        }
        return it->second;
    };

    while (true) {
        int64_t best_rank = std::numeric_limits<int64_t>::max();
        size_t best_idx = word_tokens.size();
        for (size_t i = 0; i + 1 < word_tokens.size(); ++i) {
            int64_t rank = get_pair_rank(i);
            if (rank < best_rank) {
                best_rank = rank;
                best_idx = i;
            }
        }
        if (best_idx >= word_tokens.size()) break;
        word_tokens[best_idx] =
            word_tokens[best_idx] + word_tokens[best_idx + 1];
        word_tokens.erase(word_tokens.begin() + static_cast<ptrdiff_t>(best_idx) + 1);
    }
    return word_tokens;
}

std::vector<int64_t> QwenTokenizer::encode(const std::string& text,
                                              bool add_special_tokens) const {
    (void)add_special_tokens;
    std::vector<int64_t> result;

    auto segments = split_special_tokens(text);
    for (const auto& seg : segments) {
        if (seg.is_special) {
            result.push_back(seg.id);
            continue;
        }
        auto pieces = pre_tokenize(seg.text);
        for (const auto& piece : pieces) {
            std::string encoded = byte_encode(piece);
            auto tokens = bpe(encoded);
            for (const auto& tok : tokens) {
                auto it = vocab_.find(tok);
                if (it != vocab_.end()) {
                    result.push_back(it->second);
                }
            }
        }
    }
    return result;
}

std::string QwenTokenizer::decode(const std::vector<int64_t>& ids,
                                   bool skip_special_tokens) const {
    std::vector<uint8_t> bytes;
    for (int64_t id : ids) {
        if (id < 0 || id >= static_cast<int64_t>(id_to_token_.size())) {
            continue;
        }
        if (skip_special_tokens &&
            id < static_cast<int64_t>(added_token_flags_.size()) &&
            added_token_flags_[static_cast<size_t>(id)]) {
            continue;
        }
        const std::string& token = id_to_token_[static_cast<size_t>(id)];
        const auto cps = utf8_to_codepoints(token);
        for (uint32_t c : cps) {
            auto it = unicode_to_bytes_.find(static_cast<char32_t>(c));
            if (it != unicode_to_bytes_.end()) {
                bytes.push_back(it->second);
            }
        }
    }
    return bytes_to_utf8(bytes);
}

std::string QwenTokenizer::build_chat_prompt(
    const std::vector<std::pair<std::string, std::string>>& messages,
    bool add_generation_prompt, bool enable_thinking) const {
    std::string prompt;
    for (const auto& [role, content] : messages) {
        prompt += "<|im_start|>" + role + "\n" + content +
                  "<|im_end|>\n";
    }
    if (add_generation_prompt) {
        prompt += "<|im_start|>assistant\n";
        if (enable_thinking) {
            // Qwen3.5's default generation template starts an open thinking
            // block. The model emits the reasoning and closes it itself.
            prompt += "<think>\n";
        } else {
            // Match tokenizer_config.json when enable_thinking=false.
            prompt += "<think>\n\n</think>\n\n";
        }
    }
    return prompt;
}

} // namespace hybridai::tokenizer
