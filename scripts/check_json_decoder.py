import json

raw = open('E:/models/Qwen3.8-27B-FP8/tokenizer.json', 'rb').read()
# parse first string after merges
import re
m = re.search(rb'"merges":\s*\[\s*"([^"]*)"', raw)
print('group1 raw bytes:', m.group(1))
print('as latin1:', m.group(1).decode('latin1'))
print('as utf-8:', m.group(1).decode('utf-8', errors='replace'))
# The Python json decoder should decode as unicode. Why does it produce 0x81d3?
# Let's explicitly decode raw bytes with json.loads specifying ensure_ascii? Not applicable.
# Maybe the terminal font is mapping U+0120 to 臓? But ord returned 0x81d3, so it's actually wrong char.
# Test small string:
s = b'"\\u0120"'
print('\\u0120 parsed:', json.loads(s))
# Test raw utf8
s2 = b'"\xc4\xa0"'
print('utf8 parsed:', json.loads(s2), hex(ord(json.loads(s2))))
