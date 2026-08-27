import json
from tokenizers import Tokenizer

# Build the actual byte-to-unicode map used by the Qwen3 tokenizer.
# We use the vocab: for each byte b, the single-char token at a specific id is the encoded form.
# But ids are not byte values. Instead, we can derive the map by decoding token strings:
# - token string is the byte-level encoded form.
# - decode(token_id) converts it back to the original byte.
# So if we find the token whose decode result is a single byte b, that token's string is the encoded form of b.

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

b2u_str = {}
for tok_str, tid in vocab.items():
    if len(tok_str) == 1:
        decoded = tok.decode([tid])
        if len(decoded) == 1:
            b = decoded[0].encode('latin-1')[0]
            b2u_str[b] = tok_str

print('Found mappings:', len(b2u_str))
# Print non-identity for ASCII range
for b in range(256):
    if b in b2u_str:
        s = b2u_str[b]
        if ord(s) != b:
            print(f'0x{b:02X} -> {repr(s)} ord={hex(ord(s))}')
    else:
        print(f'0x{b:02X} -> MISSING')
