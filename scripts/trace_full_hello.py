import json
from tokenizers import Tokenizer

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
merge_rank = {tuple(m.split(' ')): i for i, m in enumerate(merges)}

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')

# Encode ' how' and trace BPE
for piece in [' how']:
    enc = tok.encode(piece)
    print('piece:', repr(piece), 'ids:', enc.ids, 'tokens:', enc.tokens)
    # The first char is space (0x20). Byte encode -> 'Ġ'
    # Then BPE on ['Ġ','h','o','w']
    word = ['Ġ','h','o','w']
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
            print('no merge found at', word)
            break
        print(f'merge {best}: {repr(word[best])}+{repr(word[best+1])} rank={bestrank}')
        word[best] = word[best] + word[best+1]
        del word[best+1]
    print('final', word)
