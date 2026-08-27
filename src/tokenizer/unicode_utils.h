#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace hybridai::tokenizer {

// Decode a UTF-8 string into a vector of Unicode code points.
std::vector<uint32_t> utf8_to_codepoints(const std::string& s);

// Encode a vector of Unicode code points into a UTF-8 string.
std::string codepoints_to_utf8(const std::vector<uint32_t>& cps);

// Return the number of bytes in the UTF-8 sequence that starts with byte c.
size_t utf8_sequence_length(uint8_t c);

// Minimal codepoint category flags used by the Qwen3 pre-tokenizer.
struct CodepointFlags {
    bool is_letter = false;      // \p{L}
    bool is_number = false;      // \p{N}
    bool is_mark = false;        // \p{M}
    bool is_whitespace = false;  // \s
    bool is_defined = false;     // any recognized category
};

CodepointFlags codepoint_flags(uint32_t cp);

} // namespace hybridai::tokenizer
