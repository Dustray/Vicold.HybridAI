import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
# Find merges where first char is the space replacement char. But the reference tokenizer's space replacement is 'Ġ' U+0120.
# Search for merges where left token's first char codepoint is U+0120.
found = []
for i, m in enumerate(merges):
    a, b = m.split(' ')
    if a and ord(a[0]) == 0x120:
        found.append((i, m))
print('count', len(found))
for i, m in found[:30]:
    print(i, repr(m))
