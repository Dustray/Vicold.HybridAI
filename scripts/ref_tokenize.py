from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("E:/models/Qwen3.8-27B-FP8", trust_remote_code=True, local_files_only=True)

texts = [
    "Hello, how are you?",
    "hello world",
    "  hello   world  ",
    "2025年",
    "<|im_start|>user\nHello<|im_end|>\n",
    "'s 't 're 've 'm 'll 'd",
]

for t in texts:
    print(repr(t), tok.encode(t, add_special_tokens=False))
