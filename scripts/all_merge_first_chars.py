import json
from collections import Counter

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
c = Counter()
for m in merges[:2000]:
    a,b = m.split(' ')
    c[a[0] if a else ''] += 1
print(c.most_common(20))
