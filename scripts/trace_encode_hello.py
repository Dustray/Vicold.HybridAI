import json
from tokenizers import Tokenizer

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')

# Trace the byte-level encoding of 'Hello' using the tokenizer's normalizer+pre_tokenizer? 
# Instead, we know pre-tokenizer gives 'Hello', then ByteLevel encodes bytes of 'Hello' using the byte table.
# Let's reconstruct the string the BPE sees.
b2u = {}
for b in range(256):
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    b2u[b] = enc.tokens

def byte_encode(text):
    out = []
    for c in text.encode('utf-8'):
        out.extend(b2u[c])
    return ''.join(out)

for piece in ['Hello', ',', ' how', ' are', ' you', '?']:
    encoded = byte_encode(piece)
    print(repr(piece), '-> byte-encoded:', repr(encoded))
    enc = tok.encode(piece)
    print('  ids:', enc.ids, 'tokens:', enc.tokens)
