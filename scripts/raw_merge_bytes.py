import json
# Load raw bytes from tokenizer.json and print hex of first merge entry.
raw = open('E:/models/Qwen3.8-27B-FP8/tokenizer.json', 'rb').read()
# find the start of merges array after the key
key = b'"merges"'
pos = raw.find(key)
print('pos', pos)
# print next 200 bytes
print(raw[pos:pos+200])
# Locate first string entry after key: look for ", "
start = raw.find(b'[', pos)
print('start', start)
print(raw[start:start+300])
