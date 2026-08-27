import json

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))

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
u2b = {v: k for k, v in b2u.items()}

for m in data['model']['merges'][:20]:
    a, b = m.split(' ')
    def decompose(s):
        out = []
        for c in s:
            if c in u2b:
                out.append(f'{u2b[c]:02X}')
            else:
                out.append(f'U+{ord(c):04X}?')
        return '+'.join(out)
    print(repr(m), '=>', decompose(a), '|', decompose(b))
