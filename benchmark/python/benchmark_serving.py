#!/usr/bin/env python3
"""Serving benchmark placeholder until a HybridAI service adapter exists."""
import json

print(json.dumps({
    "module": "serving",
    "status": "skipped",
    "reason": "no_serving_api",
}, ensure_ascii=False))
