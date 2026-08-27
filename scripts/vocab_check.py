import json
tj = json.load(open("E:/models/Qwen3.8-27B-FP8/tokenizer.json", encoding="utf-8"))
vocab = tj["model"]["vocab"]
merges = tj["model"]["merges"]
print("vocab size:", len(vocab))
print("merges size:", len(merges))
# print some token examples
for token in ["Hello", ",", "Ġhow", "Ġare", "Ġyou", "?", "hello", "Ġworld", "Ġ", "ĠĠ"]:
    print(repr(token), vocab.get(token, "NOT FOUND"))
