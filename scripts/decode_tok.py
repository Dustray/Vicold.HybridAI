import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

# The reference tokenizer seems to use CJK chars for byte replacement, not the GPT-2 Latin Extended-B block.
# Find which char corresponds to byte 0x20 (space) and 0x00 in the vocab.
for tok, id in vocab.items():
    if id in [220, 188]:
        print(f'id {id}: token {repr(tok)} ord={hex(ord(tok))}')

# Decode the first 10 single-char tokens with id < 256? Not straightforward.
# Instead, use the tokenizer to encode single bytes and see the token.
from tokenizers import Tokenizer
tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
for b in range(256):
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    if len(enc.tokens) == 1:
        t = enc.tokens[0]
        if ord(t) != b:
            print(f'0x{b:02X} -> {repr(t)} ord={hex(ord(t))}')
