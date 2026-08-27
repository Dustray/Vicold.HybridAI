# Our C++ byte_encode maps bytes to char32_t and then codepoints_to_utf8.
# For byte 0x20, our mapping currently gives cp=0x120 -> UTF-8 bytes C4 A0 -> character 'Ġ' (U+0120).
# The tokenizer library's token string for id 220 is '臓' (U+81D3).
# But when encoding space, tokenizers returns tokens=['Ġ'].
# Why the discrepancy? Because tokenizers renders the token by decoding the byte-level form.
# The actual vocab entry is '臓', and decode([220]) -> ' '.
# So the BPE operates on '臓' as the encoded form of space, not on 'Ġ'.

# Wait: but our simulate_bpe_hello.py used 'Hello' (ASCII) and worked with merge ranks from tokenizer.json.
# The merges in tokenizer.json are strings like '臓 臓' and 'i n', not 'Ġ' and 'h'.
# This means the BPE operates on the raw byte-level encoded forms (the vocab token strings), not the decoded display forms.

# For ASCII letters, the encoded form equals the original char, so our code's 'Hello' is correct.
# But for space, the encoded form should be '臓' (U+81D3), not 'Ġ' (U+0120).
# Our current byte_encode for space produces 'Ġ', which is not in the merges, so ' how' fails.

# Let's verify: is '臓 h' a merge? 
import json
data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
merges = data['model']['merges']
print('臓 h in merges:', '臓 h' in merges)
print('臓h o in merges:', '臓h o' in merges)
print('臓ho w in merges:', '臓ho w' in merges)

# But also the pre-tokenizer regex splits on whitespace and outputs ' how' as a piece.
# In BPE, the first char of ' how' is space byte 0x20, encoded as '臓'.
# So the input to BPE is ['臓','h','o','w'].
# Let's simulate with this.
merge_rank = {tuple(m.split(' ')): i for i, m in enumerate(merges)}
word = ['臓','h','o','w']
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
        print('no merge at', word)
        break
    print(f'merge {best}: {repr(word[best])}+{repr(word[best+1])} rank={bestrank}')
    word[best] = word[best] + word[best+1]
    del word[best+1]
print('final', word)
