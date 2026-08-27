from tokenizers import Tokenizer
import json

def bytes_to_unicode():
    """Copied from transformers GPT-2 tokenizer."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

b2u = bytes_to_unicode()
u2b = {v: k for k, v in b2u.items()}

def byte_level_encode(s):
    return "".join(b2u[b] for b in s.encode("utf-8"))

# Load vocab and check '年'
tj = json.load(open("E:/models/Qwen3.8-27B-FP8/tokenizer.json", encoding="utf-8"))
vocab = tj["model"]["vocab"]
encoded_year = byte_level_encode("年")
print("encoded '年':", repr(encoded_year))
print("in vocab:", vocab.get(encoded_year, "NOT FOUND"))

# Show what tokenizer does for "2025年"
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
enc = t.encode("2025年")
print("ids:", enc.ids)
print("tokens:", enc.tokens)
for tok in enc.tokens:
    print(" token:", repr(tok), "id:", vocab.get(tok, "NOT FOUND"))
    # Decode the token back to bytes
    bytes_ = bytes(u2b[c] for c in tok)
    print("   bytes:", bytes_.hex(), "decoded:", bytes_.decode("utf-8", errors="replace"))
