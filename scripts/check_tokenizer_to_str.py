from tokenizers import Tokenizer

tok = Tokenizer.from_file('E:/models/Qwen3.8-27B-FP8/tokenizer.json')
print(tok.to_str()[:2000])
