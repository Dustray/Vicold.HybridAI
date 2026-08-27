import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')

b2u = {}
for b in range(256):
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    b2u[b] = enc.tokens

# Print as a compact table
import sys
out = []
for b in range(256):
    toks = b2u[b]
    out.append(f'{b:02X} | ' + repr(toks))
with open('d:/Vicold/Vicold.HybridAI/scripts/byte_table.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
