with open('E:/models/Qwen3.8-27B-FP8/tokenizer.json', 'rb') as f:
    raw = f.read()

idx = raw.find(b'"merges"')
print(raw[idx:idx+300])
