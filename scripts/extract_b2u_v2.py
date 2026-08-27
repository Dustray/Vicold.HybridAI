import json
from tokenizers import Tokenizer

# Build byte -> encoded-string mapping by encoding every byte.
# The encoded string may be multiple characters for high bytes (>=0x80).

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')

b2u = {}
for b in range(256):
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    b2u[b] = enc.tokens  # list of token strings

# Save as list of lists
with open('d:/Vicold/Vicold.HybridAI/scripts/b2u_tokens.json', 'w', encoding='utf-8') as f:
    json.dump(b2u, f, ensure_ascii=False, indent=2)

# Also save the token->byte reverse mapping (for decode)
u2b = {}
for b, toks in b2u.items():
    # For bytes that map to a single token, map that token back to byte
    if len(toks) == 1:
        u2b[toks[0]] = b
    # For multi-token bytes, we'll handle in decode by decoding token strings via a separate byte table.

with open('d:/Vicold/Vicold.HybridAI/scripts/u2b_single.json', 'w', encoding='utf-8') as f:
    json.dump({k: v for k, v in u2b.items()}, f, ensure_ascii=False, indent=2)

print('saved. single-token byte count:', len(u2b))
