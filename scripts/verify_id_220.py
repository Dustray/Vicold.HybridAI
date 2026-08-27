from tokenizers import Tokenizer

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
# Encode just space and check id
enc = tok.encode(' ')
print('space ids:', enc.ids, 'tokens:', enc.tokens)

# Encode string with single space replacement char 'Ġ' (U+0120)
enc2 = tok.encode('\u0120')
print('U+0120 ids:', enc2.ids, 'tokens:', enc2.tokens)

# Encode the CJK char used as space replacement
enc3 = tok.encode('臓')
print('臓 ids:', enc3.ids, 'tokens:', enc3.tokens)
