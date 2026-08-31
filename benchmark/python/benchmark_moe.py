#!/usr/bin/env python3
"""MoE benchmark placeholder for dense models without routing telemetry."""
import json

print(json.dumps({
    "module": "moe",
    "status": "skipped",
    "reason": "model_is_dense_or_routing_unavailable",
}, ensure_ascii=False))
