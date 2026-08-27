import json

raw = open('E:/models/Qwen3.8-27B-FP8/tokenizer.json', 'rb').read()
data = json.loads(raw.decode('utf-8'))
m = data['model']['merges'][0]
print('after utf8 decode load:', repr(m), [hex(ord(c)) for c in m])
