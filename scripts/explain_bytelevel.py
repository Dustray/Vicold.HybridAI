import json
from tokenizers import Tokenizer

# The reference tokenizer uses a different byte-to-unicode scheme than GPT-2.
# For high bytes (>=0x80), it encodes them as two bytes: first 0xC2 or 0xC3 etc, then maps each via b2u.
# So byte_encode for the reference is: treat raw byte as latin-1 char, UTF-8 encode it, then byte-level encode each UTF-8 byte.
# This is exactly the GPT-2 byte_encoder applied to the UTF-8 bytes of the pre-tokenized piece.

# Verify with a multi-byte UTF-8 char like '年' (E5 B9 B4)
tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
enc = tok.encode('年')
print('年 ids:', enc.ids)
print('年 tokens:', enc.tokens)

# Reconstruct manually:
# pre-tokenize -> '年'
# raw bytes of '年' -> b'\xe5\xb9\xb4'
# byte_encode each byte via GPT-2 mapping (but using this tokenizer's mapping)
# 0xE5 -> ???
# Let's see what tokenizers gives for each byte:
for b in [0xE5, 0xB9, 0xB4]:
    e = tok.encode(bytes([b]).decode('latin-1'))
    print(f'0x{b:02X} -> {e.tokens}')
