import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
vocab = data['model']['vocab']

# build bytes_to_unicode like GPT2
def bytes_to_unicode():
    bs = list(range(ord('!'), ord('~')+1)) + list(range(ord('¡'), ord('¬')+1)) + list(range(ord('®'), ord('ÿ')+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(c) for c in cs]
    return dict(zip(bs, cs))

b2u = bytes_to_unicode()

for m in data['model']['merges'][:20]:
    a, b = m.split(' ')
    # They are space-separated token strings (already unicode chars), not byte values.
    # Show the first byte each maps to.
    print(repr(m), '=>', repr(a), repr(b))
