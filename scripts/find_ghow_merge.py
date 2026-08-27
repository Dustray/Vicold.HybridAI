import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
for i, m in enumerate(merges):
    if 'Ġ' in m and 'h' in m and 'o' in m and 'w' in m:
        # Could be Ġhow split as different pairs
        pass
    a,b = m.split(' ')
    if a == 'Ġ' and b.startswith('h'):
        print(i, repr(m))
        if i > 100:
            break
