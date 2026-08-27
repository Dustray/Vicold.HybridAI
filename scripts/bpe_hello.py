from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers import Regex
import json

# Load vocab and merges
tj = json.load(open("E:/models/Qwen3.8-27B-FP8/tokenizer.json", encoding="utf-8"))
vocab = tj["model"]["vocab"]
merges = tj["model"]["merges"]

# Build BPE model with empty vocab/merges so encode does only pre-tokenization
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
regex_only = Sequence([Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior="isolated")])

for piece in ["Hello", ",", " how", " are", " you", "?"]:
    enc = t.encode(piece)
    print(repr(piece), "->", enc.ids, enc.tokens)
