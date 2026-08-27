import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

# Find token with id 220
for tok, id in vocab.items():
    if id == 220:
        print('id 220 token:', repr(tok), 'ord=', hex(ord(tok)))
        break

# Also check id 188 (0x00 byte)
for tok, id in vocab.items():
    if id == 188:
        print('id 188 token:', repr(tok), 'ord=', hex(ord(tok)))
        break
