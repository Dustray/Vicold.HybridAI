#!/usr/bin/env python3
"""Run serial token-throughput measurements for HybridAI."""
from __future__ import annotations

import argparse
import json
import os
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
    environment = os.environ.copy()
    if args.enable_mtp:
        environment["HYBRIDAI_ENABLE_MTP"] = "1"
    if args.speculative_mtp:
        environment["HYBRIDAI_ENABLE_MTP"] = "1"
        environment["HYBRIDAI_SPECULATIVE_MTP"] = "1"
    environment["HYBRIDAI_SPECULATIVE_WIDTH"] = str(args.speculative_width)
    try:
        completed = subprocess.run(
            command, check=True, capture_output=True, text=True, env=environment
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
        "mtp_mode": "speculative" if args.speculative_mtp else (
            "agreement_probe" if args.enable_mtp else "disabled"
        ),
        "records": len(records),
        "prompt_tokens": summarize(records, "prompt_tokens"),
        "generated_tokens": summarize(records, "generated_tokens"),
        "max_new_tokens": args.max_new_tokens,
        "speculative_width": args.speculative_width,
        "output_tokens_per_second": summarize(records, "output_tokens_per_second"),
        "decode_tokens_per_second": summarize(records, "decode_tokens_per_second"),
        "mtp_proposed_tokens": summarize(records, "mtp_proposed_tokens"),
        "mtp_accepted_tokens": summarize(records, "mtp_accepted_tokens"),
        "mtp_rejected_tokens": summarize(records, "mtp_rejected_tokens"),
        "mtp_correction_tokens": summarize(records, "mtp_correction_tokens"),
        "speculative_replay_tokens": summarize(
            records, "speculative_replay_tokens"
        ),
        "speculative_mtp_recovery_tokens": summarize(
            records, "speculative_mtp_recovery_tokens"
        ),
        "speculative_max_proposal_width": summarize(
            records, "speculative_max_proposal_width"
        ),
        "speculative_mtp_cache_clone_count": summarize(
            records, "speculative_mtp_cache_clone_count"
        ),
        "speculative_target_cache_clone_count": summarize(
            records, "speculative_target_cache_clone_count"
        ),
        "mtp_fallback_steps": summarize(records, "mtp_fallback_steps"),
        "mtp_acceptance_rate": summarize(records, "mtp_acceptance_rate"),
        "speculative_proposal_seconds": summarize(
            records, "speculative_proposal_seconds"
        ),
        "speculative_verification_seconds": summarize(
            records, "speculative_verification_seconds"
        ),
        "speculative_replay_seconds": summarize(
            records, "speculative_replay_seconds"
        ),
        "speculative_fallback_seconds": summarize(
            records, "speculative_fallback_seconds"
        ),
        "speculative_mtp_cache_clone_seconds": summarize(
            records, "speculative_mtp_cache_clone_seconds"
        ),
        "speculative_target_cache_clone_seconds": summarize(
            records, "speculative_target_cache_clone_seconds"
        ),
        "speculative_argmax_seconds": summarize(
            records, "speculative_argmax_seconds"
        ),
        "speculative_rounds": summarize(records, "speculative_rounds"),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    print_summary_table("HybridAI Throughput Benchmark", summary)


if __name__ == "__main__":
    main()
