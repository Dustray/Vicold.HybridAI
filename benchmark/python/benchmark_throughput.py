#!/usr/bin/env python3
"""Run serial token-throughput measurements for HybridAI."""
from __future__ import annotations

import argparse
import json
import subprocess

from common import (
    add_common_args,
    parse_records,
    print_summary_table,
    save_jsonl,
    summarize,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser)
    args = parser.parse_args()
    command = [args.runner, str(args.model), args.prompt, args.backend,
               str(args.max_new_tokens), str(args.warmup), str(args.runs)]
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except subprocess.CalledProcessError as exc:
        if exc.stdout:
            print(exc.stdout, end="")
        if exc.stderr:
            print(exc.stderr, end="", file=__import__("sys").stderr)
        raise SystemExit(
            f"benchmark runner failed with exit code {exc.returncode}"
        ) from exc
    records = parse_records(completed.stdout)
    if not records:
        raise SystemExit("runner produced no JSON records")
    save_jsonl(records, args.output)
    summary = {
        "module": "throughput",
        "mode": "serial",
        "records": len(records),
        "prompt_tokens": summarize(records, "prompt_tokens"),
        "generated_tokens": summarize(records, "generated_tokens"),
        "max_new_tokens": args.max_new_tokens,
        "output_tokens_per_second": summarize(records, "output_tokens_per_second"),
        "decode_tokens_per_second": summarize(records, "decode_tokens_per_second"),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print_summary_table("HybridAI Throughput Benchmark", summary)


if __name__ == "__main__":
    main()
