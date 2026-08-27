from tokenizers import Tokenizer
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
ids = t.encode("Hello, how are you?").ids
print(ids)
