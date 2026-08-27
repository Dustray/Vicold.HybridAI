from transformers import AutoTokenizer
import json

tok = AutoTokenizer.from_pretrained("E:/models/Qwen3.8-27B-FP8", trust_remote_code=True, local_files_only=True)

print("added_tokens_encoder:")
for k, v in tok.added_tokens_encoder.items():
    print(repr(k), v)

print("\nspecial_tokens_map:")
print(tok.special_tokens_map)

# Inspect tokenizer.json regex
tj = json.load(open("E:/models/Qwen3.8-27B-FP8/tokenizer.json", encoding="utf-8"))
print("\npre_tokenizer:")
print(json.dumps(tj["pre_tokenizer"], indent=2, ensure_ascii=False))
