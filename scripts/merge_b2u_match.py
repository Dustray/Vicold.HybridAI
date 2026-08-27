import json

# GPT-2 bytes_to_unicode
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

# Our byte_encode("Hello, how are you?") should map space (0x20) to chr(0x0100)? Let's verify.
print('space byte 0x20 ->', repr(b2u[0x20]), 'ord=', hex(ord(b2u[0x20])))
print('byte 0x00 ->', repr(b2u[0x00]), 'ord=', hex(ord(b2u[0x00])))

# The first few merges use '臓' which is not in our byte mapping.
print('臓 in u2b?', '臓' in u2b)
print('膴 in u2b?', '膴' in u2b)
print('Possible replacement chars in our range:', [c for c in map(chr, range(0x100, 0x180)) if c not in u2b][:10])

# Look at vocab for bytes tokens that map to space etc.
vocab = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))['model']['vocab']
# token for id 220?  (our earlier output had space as id 220)
print('id 220 token:', repr([k for k,v in vocab.items() if v==220][0]))
