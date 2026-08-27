import json
from tokenizers import Tokenizer

# Reconstruct bytes_to_unicode by encoding every byte value with the reference tokenizer.
tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')

b2u = {}
for b in range(256):
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    if len(enc.tokens) == 1 and len(enc.tokens[0]) == 1:
        b2u[b] = enc.tokens[0]
    else:
        # For bytes that don't map to single char, record tuple
        b2u[b] = tuple(enc.tokens)

# Save mapping
with open('d:/Vicold/Vicold.HybridAI/scripts/b2u_reference.json', 'w', encoding='utf-8') as f:
    # tuple not json serializable; convert to string
    json.dump({str(k): (v if isinstance(v, str) else ''.join(v)) for k, v in b2u.items()}, f, ensure_ascii=False, indent=2)

# Print non-identity mappings
for b in range(256):
    v = b2u[b]
    if isinstance(v, str):
        is_identity = (ord(v) == b)
        in_printable = (b >= ord('!') and b <= ord('~'))
        if not is_identity or (b < ord('!') or b > ord('~')):
            print(f'0x{b:02X} -> {repr(v)} ord={hex(ord(v))}')
    else:
        print(f'0x{b:02X} -> multi {v}')
