import json

with open('E:/models/Qwen3.8-27B-FP8/tokenizer.json', 'rb') as f:
    raw = f.read()

# Decode as UTF-8 and find the token for id 220 by searching for a string with "\u81d3"? Better load with json and print repr.
data = json.loads(raw.decode('utf-8'))
vocab = data['model']['vocab']
# Get token for id 220
tok_220 = [k for k,v in vocab.items() if v==220][0]
print('id 220 raw repr:', repr(tok_220), 'len', len(tok_220))
print('id 220 codepoints:', [hex(ord(c)) for c in tok_220])

# Also get id 188
tok_188 = [k for k,v in vocab.items() if v==188][0]
print('id 188 codepoints:', [hex(ord(c)) for c in tok_188])

# Search raw JSON bytes for id 220 token as encoded in file
# It will be encoded as UTF-8 bytes of the token string.
needle = tok_220.encode('utf-8')
print('raw UTF-8 bytes of token 220:', needle.hex())
