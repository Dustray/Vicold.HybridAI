#!/usr/bin/env python3
"""Prefix-cache benchmark placeholder until session cache support exists."""
import json

print(json.dumps({
    "module": "prefix_caching",
    "status": "skipped",
    "reason": "no_session_or_prefix_cache_api",
}, ensure_ascii=False))
