from tokenizers import Tokenizer
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
for text in ["  hello   world  ", "'s 't 're 've 'm 'll 'd", "hello world", "Hello, how are you?"]:
    enc = t.encode(text)
    print(repr(text))
    print("  ids:", enc.ids)
    print("  tokens:", enc.tokens)
