from tokenizers import Tokenizer
import json

# Our C++ implementation traces for "Hello, how are you?"
text = "Hello, how are you?"
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
enc = t.encode(text)
print("Reference ids:", enc.ids)
print("Reference tokens:", enc.tokens)

# Show each token id from our likely broken output
our_ids = [39, 68, 75, 75, 78, 11, 220, 71, 78, 86, 220, 64, 81, 68, 220, 88, 78, 84, 30]
vocab = json.load(open("E:/models/Qwen3.8-27B-FP8/tokenizer.json", encoding="utf-8"))["model"]["vocab"]
inv = {v: k for k, v in vocab.items()}
print("Our tokens:", [inv[i] for i in our_ids])
