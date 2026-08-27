import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
found = []
for i, m in enumerate(merges):
    if m.startswith('Ġ ') or m.startswith('Ġh ') or m.startswith('Ġho '):
        found.append((i, m))
print('count', len(found))
for i,m in found[:30]:
    print(i, repr(m))
