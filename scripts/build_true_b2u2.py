import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

# Strategy: for each byte b, encode the latin-1 char and get token string(s).
# The token string is the encoded form. We will save the token string list for each byte.
# This is what our C++ byte_encode should produce.

b2u = {}
for b in range(256):
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    b2u[b] = enc.tokens

# Print as JSON
import json as jsonmod
with open('d:/Vicold/Vicold.HybridAI/scripts/true_b2u.json', 'w', encoding='utf-8') as f:
    jsonmod.dump({f'{b:02X}': toks for b, toks in b2u.items()}, f, ensure_ascii=False, indent=2)

# Show space etc
for b in [0x20, 0x00, 0x0A, 0x80, 0xFF]:
    print(f'0x{b:02X} -> {b2u[b]}')
