import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
print('total merges', len(merges))
print('first 10 merges:', merges[:10])
print('last 5 merges:', merges[-5:])
# Find any merge whose token starts with a char above ASCII
for m in merges[:200]:
    a, b = m.split(' ')
    if any(ord(c) > 127 for c in a + b):
        print('non-ascii merge:', repr(m))
