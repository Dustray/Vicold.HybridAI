import json

# The Qwen3 tokenizer uses a byte-level scheme where:
# - bytes 0x00-0x20 and 0x7F map to U+0100..U+0121 (same as GPT-2 for these)
# - bytes 0x21-0x7E map to themselves
# - bytes 0x80-0xFF are represented by encoding the byte as UTF-8 (which gives 2 bytes for 0x80-0xBF and 0xC0-0xDF? Actually for 0x80-0xFF in latin-1, UTF-8 encode gives 2 bytes: 0xC2 0x80..0xBF for 0x80-0xBF, 0xC3 0x80..0xBF for 0xC0-0xFF)
#   Then each of those UTF-8 bytes is mapped using the same byte-level scheme.
# So byte_encode(byte b) = byte_encode_utf8(utf8_bytes_of_b)
# For 0x80: UTF-8 bytes are [0xC2, 0x80]. byte_encode(0xC2)=? From table: 0xC2 -> ['ÃĤ']. byte_encode(0x80)->['ÂĢ']? Wait 0x80 itself maps to ['ÂĢ'].
# But we should not recursively encode; the tokenizer's normal ByteLevel pre-tokenizer maps each raw byte to a single token string according to a fixed table.

# From dump_byte_table, for high bytes the token strings are mostly 2 chars, e.g. 0x80 -> 'ÂĢ'.
# This suggests the tokenizer uses a byte table where byte b maps to a single Unicode character.
# But for b>=0x80, the character is in the CJK range? Let's check the ord of 'Â' and 'Ģ'.
print('Â ord:', hex(ord('Â')))
print('Ģ ord:', hex(ord('Ģ')))
print('臓 ord:', hex(ord('臓')))
# 'Â' is U+00C2, 'Ģ' is U+0122. These are not CJK. Wait earlier we thought id 220 vocab entry was '臓' U+81D3.
# But tokenizers displays token as 'Ġ'. The display is the decoded byte, not the vocab string.
# The actual vocab string for id 220 must be different. Let's read tokenizer.json directly with raw bytes.
