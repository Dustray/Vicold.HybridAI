import json
from tokenizers import Tokenizer

# Build a direct byte -> encoded-char mapping by decoding vocab token strings.
# For each single-char vocab token, decode it back to bytes. The token represents a single byte.

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

b2u = {}
for tok_str, tid in vocab.items():
    if len(tok_str) == 1:
        decoded = tok.decode([tid])
        # decoded should be a single byte when interpreted as latin-1
        if len(decoded) == 1:
            try:
                byte_val = decoded.encode('latin-1')[0]
            except UnicodeEncodeError:
                continue
            b2u[byte_val] = tok_str

print('Found', len(b2u), 'single-byte mappings')
# Check completeness
missing = [b for b in range(256) if b not in b2u]
print('missing:', missing)

# Save as compact format: list of 256 strings
with open('d:/Vicold/Vicold.HybridAI/scripts/b2u_compact.json', 'w', encoding='utf-8') as f:
    json.dump([b2u.get(b, '') for b in range(256)], f, ensure_ascii=False)
