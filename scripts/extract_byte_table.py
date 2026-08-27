import json
from tokenizers import Tokenizer

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

# The byte table may be stored in tokenizer.json under "model"? Search keys.
print('model keys:', list(data['model'].keys()))

# Try to reconstruct by encoding single bytes and seeing what tokens are returned.
tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
for b in [0x00, 0x09, 0x0A, 0x0D, 0x20, 0x21, 0x41, 0x80, 0xFF, 0xC4]:
    s = bytes([b]).decode('latin-1')
    enc = tok.encode(s)
    print(f'byte 0x{b:02X} -> ids {enc.ids} tokens {enc.tokens}')
