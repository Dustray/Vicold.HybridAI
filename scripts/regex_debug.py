from tokenizers.pre_tokenizers import Sequence, Split
from tokenizers import Regex

# Use the same regex as tokenizer.json
regex_only = Sequence([Split(Regex("(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"), behavior="isolated")])

texts = [
    "Hello, how are you?",
    "hello world",
    "  hello   world  ",
    "2025年",
    "<|im_start|>user\nHello\n<|im_end|>",
    "'s 't 're 've 'm 'll 'd",
    "안녕하세요",
    "こんにちは",
    "中文测试",
    "Café",
    "naïve",
]
for t in texts:
    pieces = regex_only.pre_tokenize_str(t)
    print(repr(t), [p[0] for p in pieces])
