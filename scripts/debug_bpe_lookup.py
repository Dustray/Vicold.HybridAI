import json

# Save the first 50 merges and some specific ones to a file for C++ debugging.
data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']

out = {
    'first_50': merges[:50],
    'hello_merges': [m for m in merges if m.split(' ')[0] in ['H','e','l','o','He','el','ll','lo','Hel','ell','llo','Hell','ello','Hello']],
    'space_merges': [m for m in merges[:2000] if 'Ġ' in m][:50]
}
with open('d:/Vicold/Vicold.HybridAI/scripts/debug_merges.json', 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('saved')
