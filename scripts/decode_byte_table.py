import json
from tokenizers import Tokenizer

# The Qwen3 tokenizer's byte_to_unicode is essentially:
# - ASCII printable bytes 0x21-0x7E map to themselves
# - Control bytes 0x00-0x1F and 0x7F map to U+0100..U+0121 (GPT-2 style)
# - Bytes 0x80-0xFF map to CJK Unified Ideographs Extension A block U+8100..U+81FF? Let's verify.

# The single-char tokens we saw include codepoints from U+8100 to U+81FF and some others.
# Specifically id 220 -> U+81D3 '臓', id 188 -> U+81A7 '膧', id 198 -> U+81B4 '膴', id 197 -> U+81B2 '膲', id 201 -> U+81B7 '膷'.

# Let's build the mapping from the vocab directly: for each byte b, find the single-char token whose id equals the expected byte-level id.
# But we don't know expected ids directly. However we can infer: the byte-level representation maps byte b to a single char token.
# The ids for bytes 0x00..0xFF should be a contiguous block or based on GPT-2 offsets?

# Actually from dump_byte_table.py we have the token strings for each byte. We can build b2u from that.
# The key point: our C++ byte_encode produces wrong chars for space (it produces U+0120 'Ġ', but the vocab uses U+81D3 '臓').
# Wait, but tokenizers library reported token 'Ġ' for space with id 220. The vocab entry for id 220 is '臓'.
# How can token be 'Ġ' but vocab entry be '臓'? Tokenizer library might be rendering the token by decoding the bytes.

# Let's check: decode id 220 with tokenizer.decode
tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
print('decode [220]:', tok.decode([220]))
print('decode [188]:', tok.decode([188]))

# If decode returns space, then the vocab string '臓' is the byte-level encoded form of space, and the rendering 'Ġ' is the decoded byte.
