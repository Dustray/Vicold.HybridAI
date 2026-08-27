import json

with open('E:/models/Qwen3.8-27B-FP8/tokenizer.json','r',encoding='utf-8') as f:
    text = f.read()

# Find the merges section raw text
mstart = text.find('"merges"')
mend = text.find('"added_tokens"', mstart)
print('raw merges text start:')
print(text[mstart:mstart+300])
print('---')

data = json.loads(text)
merges = data['model']['merges']
print('parsed first merge:', repr(merges[0]))
print('codepoints:', [hex(ord(c)) for c in merges[0]])
