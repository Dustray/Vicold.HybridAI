import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

# The tokenizer.json uses different replacement characters than GPT-2.
# Find which chars correspond to byte values, by checking single-char tokens for non-printable bytes.
# For each byte 0x80-0xFF or control bytes, the vocab likely has a single char token.

# Let's find tokens that are single chars with codepoint >= 0x100 and see if they map to bytes.
byte_like = []
for tok in vocab:
    if len(tok) == 1:
        cp = ord(tok)
        if cp >= 0x100:
            byte_like.append((cp, tok))

byte_like.sort()
print('First 50 single-char tokens with cp >= 0x100:')
for cp, tok in byte_like[:50]:
    print(hex(cp), repr(tok))

print('Total such tokens:', len(byte_like))
