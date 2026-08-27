import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']

# Find merges involving space replacement char. Determine the actual Unicode char used in merges.
# Print all merges that start with the space-replacement char (whatever it is).
# We know id 220 token is 'Ġ' (U+0120). Search merges where first char of left token has cp U+0120.
found = []
for i, m in enumerate(merges):
    a, b = m.split(' ')
    if a and ord(a[0]) == 0x120:
        found.append((i, m))
print('count', len(found))
for i, m in found[:20]:
    print(i, repr(m))
