from tokenizers import Tokenizer
import json

t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
text = "Hello, how are you?"
enc = t.encode(text)
print("ids:", enc.ids)
print("tokens:", enc.tokens)

# Check a piece: 'Ġhow' -> bytes -> should be b' how'
from bytelevel_explore import bytes_to_unicode, u2b
u2b = {v: k for k, v in bytes_to_unicode().items()}
for tok in enc.tokens:
    print(repr(tok), "->", bytes(u2b[c] for c in tok).decode("utf-8", errors="replace"))
