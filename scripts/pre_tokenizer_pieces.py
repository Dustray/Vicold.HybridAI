from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers import Regex

# Load tokenizer.json and replace BPE vocab with empty so we can inspect only pre-tokenizer pieces
t = Tokenizer.from_file("E:/models/Qwen3.8-27B-FP8/tokenizer.json")
pre = t.pre_tokenizer

# Use the pre_tokenizer to pre_tokenize the string into pieces (with ByteLevel)
text = "Hello, how are you?"
pretok = pre.pre_tokenize_str(text)
print("pre_tokenized pieces:", pretok)

# Try without ByteLevel to see raw regex matches
regex_only = Sequence([Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior="isolated")])
print("regex only pieces:", regex_only.pre_tokenize_str(text))
