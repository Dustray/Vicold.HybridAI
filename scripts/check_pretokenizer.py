import json
from tokenizers import Tokenizer

data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
print(json.dumps(data['pre_tokenizer'], indent=2, ensure_ascii=False))
