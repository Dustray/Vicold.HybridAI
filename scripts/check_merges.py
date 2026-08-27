import json
data = json.load(open('E:/models/Qwen3.8-27B-FP8/tokenizer.json'))
print('merges type:', type(data['model']['merges']))
print('first 10 merges:', data['model']['merges'][:10])
print('merges count:', len(data['model']['merges']))
print('first vocab items:', list(data['model']['vocab'].items())[:5])
