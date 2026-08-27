#include "tokenizer/unicode_utils.h"

#include <array>
#include <cctype>

namespace hybridai::tokenizer {

namespace {

// Ranges of Unicode letters (Ll/Lu/Lt/Lm/Lo), marks (Mn/Mc/Me), numbers (Nd).
// Only the ranges needed for CJK and common scripts are included. Missing ranges
// are treated as undefined, which still produce correct regex split behavior
// because the Qwen3 regex falls through to the catch-all symbol branch.
struct Range {
    uint32_t first;
    uint32_t last;
    uint32_t kind; // 1=letter, 2=number, 3=mark
};

// Sourced from Unicode 15.0 derived general categories used by Qwen3 regex.
// Letter ranges include Latin, Greek, Cyrillic, CJK, Korean, Japanese.
// Mark ranges include combining diacritical marks.
// Number ranges include ASCII digits and fullwidth digits.
static constexpr std::array<Range, 59> k_ranges = {{
    // Basic Latin letters and digits
    {0x0041, 0x005A, 1}, {0x0061, 0x007A, 1},
    // Latin-1 supplement letters
    {0x00C0, 0x00D6, 1}, {0x00D8, 0x00F6, 1}, {0x00F8, 0x00FF, 1},
    // Latin Extended-A/B, IPA
    {0x0100, 0x024F, 1}, {0x0250, 0x02AF, 1},
    // Greek and Coptic
    {0x0370, 0x03FF, 1},
    // Cyrillic
    {0x0400, 0x04FF, 1}, {0x0500, 0x052F, 1},
    // Armenian, Hebrew, Arabic
    {0x0530, 0x058F, 1}, {0x0590, 0x05FF, 1}, {0x0600, 0x06FF, 1},
    // Combining Diacritical Marks
    {0x0300, 0x036F, 3},
    // Combining Diacritical Marks Extended/Supplement
    {0x1AB0, 0x1AFF, 3}, {0x1DC0, 0x1DFF, 3},
    // Devanagari, Bengali, Tamil, Telugu, Kannada
    {0x0900, 0x097F, 1}, {0x0980, 0x09FF, 1}, {0x0B80, 0x0BFF, 1},
    {0x0C00, 0x0C7F, 1}, {0x0C80, 0x0CFF, 1},
    // Thai, Lao, Tibetan
    {0x0E00, 0x0E7F, 1}, {0x0E80, 0x0EFF, 1}, {0x0F00, 0x0FFF, 1},
    // Georgian, Hangul Jamo
    {0x10A0, 0x10FF, 1}, {0x1100, 0x11FF, 1},
    // CJK Unified Ideographs Extension A/B/C/D/E/F/G/H/I
    {0x3400, 0x4DBF, 1},
    {0x4E00, 0x9FFF, 1},
    {0x20000, 0x2A6DF, 1}, {0x2A700, 0x2B73F, 1}, {0x2B740, 0x2B81F, 1},
    {0x2B820, 0x2CEAF, 1}, {0x2CEB0, 0x2EBEF, 1},
    // CJK Compatibility Ideographs
    {0xF900, 0xFAFF, 1}, {0x2F800, 0x2FA1F, 1},
    // Hangul Syllables
    {0xAC00, 0xD7AF, 1},
    // Hiragana, Katakana, Katakana Phonetic Extensions
    {0x3040, 0x309F, 1}, {0x30A0, 0x30FF, 1}, {0x31F0, 0x31FF, 1},
    // Bopomofo
    {0x3100, 0x312F, 1}, {0x31A0, 0x31BF, 1},
    // Yi, Cherokee, Ethiopic
    {0xA000, 0xA48F, 1}, {0x13A0, 0x13FF, 1}, {0x1200, 0x137F, 1},
    // Numbers
    {0x0030, 0x0039, 2}, {0x0660, 0x0669, 2}, {0x06F0, 0x06F9, 2},
    {0x07C0, 0x07C9, 2}, {0x0966, 0x096F, 2}, {0x09E6, 0x09EF, 2},
    {0x0B66, 0x0B6F, 2}, {0x0BE6, 0x0BEF, 2}, {0x0C66, 0x0C6F, 2},
    {0x0CE6, 0x0CEF, 2}, {0x0D66, 0x0D6F, 2}, {0x0E50, 0x0E59, 2},
    {0x0ED0, 0x0ED9, 2}, {0x0F20, 0x0F29, 2}, {0xFF10, 0xFF19, 2},
}};

static constexpr std::array<uint32_t, 26> k_whitespace = {{
    0x0009, 0x000A, 0x000B, 0x000C, 0x000D, 0x0020, 0x0085, 0x00A0,
    0x1680, 0x2000, 0x2001, 0x2002, 0x2003, 0x2004, 0x2005, 0x2006,
    0x2007, 0x2008, 0x2009, 0x200A, 0x2028, 0x2029, 0x202F, 0x205F,
    0x3000, 0x180E,
}};

bool is_in_ranges(uint32_t cp, uint32_t kind) {
    // Linear scan is fine for a small number of ranges.
    for (const auto& r : k_ranges) {
        if (r.kind == kind && cp >= r.first && cp <= r.last) {
            return true;
        }
    }
    return false;
}

bool is_whitespace_cpt(uint32_t cp) {
    for (uint32_t ws : k_whitespace) {
        if (cp == ws) return true;
    }
    return false;
}

} // namespace

size_t utf8_sequence_length(uint8_t c) {
    if ((c & 0x80) == 0) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1; // invalid leading byte treated as single byte
}

std::vector<uint32_t> utf8_to_codepoints(const std::string& s) {
    std::vector<uint32_t> out;
    for (size_t i = 0; i < s.size();) {
        uint8_t c = static_cast<uint8_t>(s[i]);
        uint32_t cp = 0;
        size_t len = utf8_sequence_length(c);
        if (len == 1 && (c & 0x80) != 0) {
            ++i;
            continue; // invalid leading byte
        }
        cp = c & (0xFF >> len);
        if (len == 1) {
            cp = c;
        }
        for (size_t j = 1; j < len && (i + j) < s.size(); ++j) {
            uint8_t n = static_cast<uint8_t>(s[i + j]);
            if ((n & 0xC0) != 0x80) {
                cp = 0;
                break;
            }
            cp = (cp << 6) | (n & 0x3F);
        }
        if (cp != 0) {
            out.push_back(cp);
        }
        i += len;
    }
    return out;
}

std::string codepoints_to_utf8(const std::vector<uint32_t>& cps) {
    std::string out;
    for (uint32_t cp : cps) {
        if (cp <= 0x7F) {
            out.push_back(static_cast<char>(cp));
        } else if (cp <= 0x7FF) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp <= 0xFFFF) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
    }
    return out;
}

CodepointFlags codepoint_flags(uint32_t cp) {
    CodepointFlags f;
    // Treat all valid non-surrogate Unicode codepoints as "defined" for the
    // Qwen3 regex, even when they are not letter/number/mark/whitespace.
    // This lets punctuation and symbols enter the symbol branch of the
    // pre-tokenizer instead of falling through to the catch-all.
    f.is_defined = (cp < 0x110000) && (cp < 0xD800 || cp > 0xDFFF);
    if (is_whitespace_cpt(cp)) {
        f.is_whitespace = true;
        return f;
    }
    if (is_in_ranges(cp, 1)) {
        f.is_letter = true;
        return f;
    }
    if (is_in_ranges(cp, 2)) {
        f.is_number = true;
        return f;
    }
    if (is_in_ranges(cp, 3)) {
        f.is_mark = true;
        return f;
    }
    return f;
}

} // namespace hybridai::tokenizer
