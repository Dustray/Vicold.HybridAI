import json
from tokenizers import Tokenizer

# The key insight: Qwen3 tokenizer's byte-level mapping is NOT GPT-2's.
# It maps each byte b to a single Unicode char in the BMP Private Use Area / extended latin range,
# but for bytes >= 0x80 it uses multi-char sequences that correspond to UTF-8-encoding the byte as latin-1 and then byte-encoding each UTF-8 byte.
# However the vocab stores single merged tokens for many byte combinations, so byte_encode does not need to output single chars for high bytes.
# The merges operate on those encoded strings.

# Let's simulate with the tokenizer library step by step to see what the BPE input looks like for "Hello".
tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')

# Get the pre-tokenizer output
from tokenizers.pre_tokenizers import ByteLevel

# Actually we can use the tokenizer's normalizer+pre_tokenizer manually if needed.
# But simpler: use the reference output tokens for 'Hello' which are ['Hello'].
# That means the BPE merge sequence reduced the byte-encoded form of 'Hello' to 'Hello'.

# Our current C++ code's byte_encode gives H/e/l/l/o because ASCII bytes map 1:1.
# The merge 'H e' exists (rank 1209). Why doesn't our BPE merge them?
# Let's check the actual encoded string our code produces for 'Hello'.

# In C++ byte_encode:
# for each byte b: codepoints_to_utf8({b}) for ASCII is just chr(b). Correct.
# So word_tokens = ['H','e','l','l','o'].
# merges_ key for ('H','e') should exist. Let's verify Python side exact strings.
merges = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))['model']['merges']
print('H e index:', merges.index('H e'))
print('e l index:', merges.index('e l') if 'e l' in merges else 'NO')
print('He l index:', merges.index('He l') if 'He l' in merges else 'NO')
print('Hel l index:', merges.index('Hel l') if 'Hel l' in merges else 'NO')
print('Hell o index:', merges.index('Hell o') if 'Hell o' in merges else 'NO')

# If all exist, then BPE should merge. Why doesn't C++ find them?
# Possible issue: PairHash or map key construction using std::string.
# Let's print the actual merge list around rank 1200 to see format.
print('merges[1200:1220]:', merges[1200:1220])
