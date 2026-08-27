from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("E:/models/Qwen3.8-27B-FP8", trust_remote_code=True, local_files_only=True)
ids = [9419, 11, 1204, 513, 488, 30]
print("decode:", repr(tok.decode(ids, skip_special_tokens=False)))
