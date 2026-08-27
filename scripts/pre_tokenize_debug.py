from tokenizers import Tokenizer, pre_tokenizers, decoders

t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
# Replace BPE model with unknown to inspect pre-tokenizer output only
# Actually we can use the pre_tokenize method if available on PreTokenizedString
# Let's just encode and show the tokens.
text = "Hello, how are you?"
encoding = t.encode(text, add_special_tokens=False)
print("ids:", encoding.ids)
print("tokens:", encoding.tokens)
print("words:", encoding.words)

# Show offset mapping
for (token, offsets) in zip(encoding.tokens, encoding.offsets):
    print(repr(token), offsets)
