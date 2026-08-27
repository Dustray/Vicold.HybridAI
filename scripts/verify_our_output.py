import json
from tokenizers import Tokenizer

# Why does our C++ output tokens look like ['H','e','l','l','o',',','Ġ','h','o','w',...]?
# That means our C++ byte_encode for space produced 'Ġ' (U+0120) and the BPE did not merge because merges use 'Ġ'.
# But earlier we saw merges use '臓'? That was a display artifact of some script/output. The actual merges use 'Ġ'.
# Let's verify the BPE with 'Ġ'.

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
print('Ġ h in merges:', 'Ġ h' in merges)
print('Ġh o in merges:', 'Ġh o' in merges)
print('Ġho w in merges:', 'Ġho w' in merges)

merge_rank = {tuple(m.split(' ')): i for i, m in enumerate(merges)}
word = ['Ġ','h','o','w']
while len(word) >= 2:
    best = None
    bestrank = 1e9
    for i in range(len(word)-1):
        pair = (word[i], word[i+1])
        r = merge_rank.get(pair)
        if r is not None and r < bestrank:
            bestrank = r
            best = i
    if best is None:
        print('no merge at', word)
        break
    print(f'merge {best}: {repr(word[best])}+{repr(word[best+1])} rank={bestrank}')
    word[best] = word[best] + word[best+1]
    del word[best+1]
print('final', word)
