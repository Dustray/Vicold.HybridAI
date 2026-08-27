from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers import Regex

text = "Hello, how are you?"
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")

# Just regex, no byte level
regex_only = Sequence([Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior="isolated")])
print("regex pieces:", regex_only.pre_tokenize_str(text))

enc = t.encode(text)
print("ids:", enc.ids)
print("tokens:", enc.tokens)
