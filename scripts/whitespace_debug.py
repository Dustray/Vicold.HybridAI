from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Sequence, Split, ByteLevel
from tokenizers import Regex

regex_only = Sequence([Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior="isolated")])

for text in ["  hello   world  ", "hello world", "'s 't 're 've 'm 'll 'd"]:
    print(repr(text), regex_only.pre_tokenize_str(text))
