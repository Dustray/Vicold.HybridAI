import json, unicodedata

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
m = merges[0]
print('repr:', repr(m))
print('len chars:', len(m))
for i, c in enumerate(m):
    print(i, repr(c), hex(ord(c)), c.encode('utf-8').hex(), unicodedata.name(c, '<none>'))
