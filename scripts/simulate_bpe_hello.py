import json
from tokenizers import Tokenizer

merges = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))['model']['merges']
vocab = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))['model']['vocab']
merge_rank = {tuple(m.split(' ')): i for i, m in enumerate(merges)}

word = list('Hello')  # byte-encoded form for ASCII is just chars
print('start', word)
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
        break
    print(f'merge {best}: {word[best]}+{word[best+1]} rank={bestrank}')
    word[best] = word[best] + word[best+1]
    del word[best+1]

print('final', word)
print('id', vocab.get(word[0]))
