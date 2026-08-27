import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']

# Save a small JSON file with specific merge pairs for C++ to load and verify.
out = {
    'test_pairs': [
        ('H', 'e'),
        ('e', 'l'),
        ('el', 'lo'),
        ('H', 'ello'),
        ('Ġ', 'h'),
        ('Ġh', 'o'),
        ('Ġho', 'w'),
    ],
    'ranks': []
}
for a,b in out['test_pairs']:
    s = f'{a} {b}'
    try:
        out['ranks'].append(merges.index(s))
    except ValueError:
        out['ranks'].append(-1)

with open('d:/Vicold/Vicold.HybridAI/scripts/test_pairs.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(out)
