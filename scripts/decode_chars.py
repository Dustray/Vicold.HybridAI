import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
m = merges[0]
# Determine actual char for space replacement. Print codepoints.
print('merge0', repr(m), [hex(ord(c)) for c in m])
# The visible char is 0x81d3? Let's list unique non-ascii chars that appear and their frequency
from collections import Counter
chars = Counter()
for m in merges[:1000]:
    for c in m:
        if ord(c) > 127:
            chars[c] += 1
print('top non-ascii chars:', chars.most_common(20))
# For each top char, print bytes of its UTF-8 encoding
for ch, cnt in chars.most_common(5):
    print('char', repr(ch), 'cp', hex(ord(ch)), 'bytes', ch.encode('utf-8').hex())
