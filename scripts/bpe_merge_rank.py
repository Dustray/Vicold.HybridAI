import json
from tokenizers import Tokenizer

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']

# build vocab string->id
vocab = data['model']['vocab']

# Check whether some common merge pairs exist in the merge list
test_pairs = [('H','e'),('He','l'),('Hel','l'),('Hell','o'),('o',','),('Ġ','h'),('Ġh','o'),('Ġho','w')]
for a,b in test_pairs:
    s = f"{a} {b}"
    try:
        idx = merges.index(s)
        print(repr(s), 'rank', idx)
    except ValueError:
        print(repr(s), 'NOT IN MERGES')

print('vocab Hello id:', vocab.get('Hello'))
print('vocab Ġhow id:', vocab.get('Ġhow'))
