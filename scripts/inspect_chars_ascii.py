import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

ids = [220, 188, 198, 197, 201]
for tok, id in vocab.items():
    if id in ids:
        print(f'id {id}: token={repr(tok)} len={len(tok)}')

# Count tokens that are single char and their codepoints
codepoints = {}
for tok in vocab:
    if len(tok) == 1:
        codepoints[ord(tok)] = tok

print('Single-char tokens count:', len(codepoints))
# Print sorted by codepoint for lower range
with open('d:/Vicold/Vicold.HybridAI/scripts/single_chars_ascii.txt', 'w', encoding='utf-8') as f:
    for cp in sorted(codepoints):
        f.write(f'{cp:04X} {repr(codepoints[cp])}\n')
print('done')
