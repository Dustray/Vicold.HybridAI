import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']

# Search for merges containing the space replacement char anywhere.
# Use the actual char from vocab id 220.
space_char = 'Ġ'
found = [m for m in merges if space_char in m]
print('total merges with Ġ:', len(found))
print('first 20:', found[:20])

# Maybe the space replacement in merges is a different char that looks same? Let's check codepoints of all chars in first few found.
if found:
    m = found[0]
    print('codepoints of first:', [hex(ord(c)) for c in m])
