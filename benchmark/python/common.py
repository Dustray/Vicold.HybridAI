#!/usr/bin/env python3
"""Shared helpers for HybridAI benchmark runners."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Iterable


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runner", required=True, help="C++ benchmark runner")
    parser.add_argument("--model", required=True, help="Model directory")
    parser.add_argument("--prompt", default="Hello, how are you? And who are you?")
    parser.add_argument("--backend", default="hip")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("benchmark.jsonl"))
    parser.add_argument(
        "--enable-mtp",
        action="store_true",
        help="Enable Qwen native MTP agreement probing in the runner",
    )
    parser.add_argument(
        "--speculative-mtp",
        action="store_true",
        help="Enable fixed-width multi-token greedy speculative MTP verification",
    )
    parser.add_argument(
        "--speculative-width",
        type=int,
        choices=range(1, 17),
        metavar="1..16",
        default=8,
        help="MTP proposal width (forwarded as HYBRIDAI_SPECULATIVE_WIDTH)",
    )


def parse_records(text: str) -> list[dict]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            records.append(json.loads(line))
    return records


def percentile(values: Iterable[float], quantile: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def summarize(records: list[dict], metric: str) -> dict:
    values = [float(record[metric]) for record in records if metric in record]
    return {
        "samples": len(values),
        "mean": statistics.fmean(values) if values else None,
        "min": min(values) if values else None,
        "max": max(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def save_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records))


def print_summary_table(title: str, summary: dict) -> None:
    statistic_columns = ("samples", "mean", "min", "max", "p50", "p90", "p95", "p99")
    metric_rows: list[tuple[str, dict]] = []
    metadata_rows: list[tuple[str, str]] = []
    for name, value in summary.items():
        if isinstance(value, dict):
            metric_rows.append((name, value))
        else:
            metadata_rows.append((format_metric_name(name), str(value)))

    metric_width = max(
        [len("Metric")] + [len(format_metric_name(name)) for name, _ in metric_rows]
    )
    column_widths = {
        column: max(
            len(column),
            max(
                (
                    format_statistic(values.get(column), column, name)
                    for name, values in metric_rows
                ),
                default="",
            ).__len__(),
        )
        for column in statistic_columns
    }

    def separator_line() -> str:
        fields = ["-" * (metric_width + 2)] + [
            "-" * (column_widths[column] + 2) for column in statistic_columns
        ]
        return "+" + "+".join(fields) + "+"

    def print_table_line(line: str) -> None:
        print(f"  {line}")

    print(f"\n{title}")
    if metadata_rows:
        for name, value in metadata_rows:
            print(f"  {name}: {value}")
    print_table_line(separator_line())
    header = [f"{'Metric':<{metric_width}}"] + [
        f"{column:>{column_widths[column]}}" for column in statistic_columns
    ]
    print_table_line("| " + " | ".join(header) + " |")
    print_table_line(separator_line())
    for name, values in metric_rows:
        display_name = format_metric_name(name)
        fields = [f"{display_name:<{metric_width}}"] + [
            f"{format_statistic(values.get(column), column, name):>{column_widths[column]}}"
            for column in statistic_columns
        ]
        print_table_line("| " + " | ".join(fields) + " |")
    print_table_line(separator_line())


def format_statistic(
    value: float | int | None, column: str, metric_name: str | None = None
) -> str:
    if value is None:
        return "N/A"
    if column == "samples" or metric_name in {
        "prompt_tokens",
        "generated_tokens",
        "mtp_proposed_tokens",
        "mtp_accepted_tokens",
        "mtp_rejected_tokens",
        "mtp_correction_tokens",
        "speculative_replay_tokens",
        "speculative_mtp_recovery_tokens",
        "speculative_max_proposal_width",
        "speculative_mtp_cache_clone_count",
        "speculative_target_cache_clone_count",
        "mtp_fallback_steps",
        "speculative_rounds",
    }:
        return str(int(value))
    if metric_name == "mtp_acceptance_rate":
        return f"{float(value) * 100:.0f}%"
    if metric_name is not None and metric_name.endswith("seconds"):
        return f"{float(value) * 1000:.3f}"
    if metric_name is not None and metric_name.endswith("tokens_per_second"):
        return f"{float(value):.2f}"
    return f"{float(value):.6f}"


def format_metric_name(name: str, statistic: str | None = None) -> str:
    metric = name if statistic is None else f"{name}.{statistic}"
    if name in {
        "prompt_tokens",
        "generated_tokens",
        "max_new_tokens",
        "mtp_proposed_tokens",
        "mtp_accepted_tokens",
        "mtp_rejected_tokens",
        "mtp_correction_tokens",
        "speculative_replay_tokens",
        "speculative_mtp_recovery_tokens",
        "speculative_max_proposal_width",
        "speculative_mtp_cache_clone_count",
        "speculative_target_cache_clone_count",
        "mtp_fallback_steps",
        "speculative_rounds",
    }:
        unit = "token"
    elif name == "mtp_acceptance_rate":
        unit = ""
    elif name.endswith("tokens_per_second"):
        unit = "token/s"
    elif name.endswith("seconds"):
        unit = "ms"
    elif name in {"records", "samples"}:
        unit = "count"
    else:
        unit = ""
    return f"{metric} [{unit}]" if unit else metric
