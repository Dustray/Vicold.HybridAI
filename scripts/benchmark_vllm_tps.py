#!/usr/bin/env python3
"""Benchmark the official vLLM engine with the same request as qwen_infer.

The benchmark deliberately builds and verifies the Qwen chat prompt, then sends
its token IDs to vLLM. This avoids silently benchmarking a different chat
template.

Example (single GPU):
    HIP_VISIBLE_DEVICES=0 /usr/bin/python3 scripts/benchmark_vllm_tps.py \
            --tensor-parallel-size 1

Example (eight GPUs):
    HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 /usr/bin/python3 \
            scripts/benchmark_vllm_tps.py --tensor-parallel-size 8
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import statistics
import time
from dataclasses import dataclass
from typing import Any, Sequence

MODEL_DEFAULT = (
    "/public/home/panyq/yiny/modelscope/models/"
    "Qwen--Qwen3.8-27B/snapshots/master"
)
USER_TEXT = "Hello, how are you?"
EXPECTED_PROMPT_IDS = [
    248045,
    846,
    198,
    9419,
    11,
    1204,
    513,
    488,
    30,
    248046,
    198,
    248045,
    74455,
    198,
    248068,
    271,
    248069,
    271,
]


@dataclass(frozen=True)
class RunResult:
    run: int
    prompt_tokens: int
    generated_tokens: int
    elapsed_seconds: float
    e2e_tps: float
    decode_tokens: int | None
    decode_seconds: float | None
    decode_tps: float | None
    text: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure official vLLM throughput using qwen_infer's exact chat "
            "prompt, greedy decoding, and generation limit."
        )
    )
    parser.add_argument("--model", default=MODEL_DEFAULT, help="Model directory")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used by vLLM (default: 1)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum generated tokens, matching qwen_infer (default: 100)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM maximum sequence length (default: 4096)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory available to vLLM (default: 0.90)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Measured requests after model loading (default: 1)",
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=0,
        help="Unmeasured warmup requests after model loading (default: 0)",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA/HIP graph execution in vLLM",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow model repository Python code",
    )
    args = parser.parse_args()

    if args.tensor_parallel_size <= 0:
        parser.error("--tensor-parallel-size must be positive")
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.max_model_len <= len(EXPECTED_PROMPT_IDS):
        parser.error("--max-model-len must exceed the prompt length")
    if not 0.0 < args.gpu_memory_utilization <= 1.0:
        parser.error("--gpu-memory-utilization must be in (0, 1]")
    if args.runs <= 0:
        parser.error("--runs must be positive")
    if args.warmup_runs < 0:
        parser.error("--warmup-runs cannot be negative")
    return args


def build_prompt_ids(tokenizer: Any) -> list[int]:
    messages = [{"role": "user", "content": USER_TEXT}]
    prompt_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if hasattr(prompt_ids, "tolist"):
        prompt_ids = prompt_ids.tolist()
    if prompt_ids and isinstance(prompt_ids[0], list):
        prompt_ids = prompt_ids[0]
    prompt_ids = [int(token_id) for token_id in prompt_ids]

    if prompt_ids != EXPECTED_PROMPT_IDS:
        raise RuntimeError(
            "Chat-template token IDs differ from qwen_infer.\n"
            f"Expected: {EXPECTED_PROMPT_IDS}\n"
            f"Actual:   {prompt_ids}\n"
            "Use the same model snapshot/tokenizer before comparing TPS."
        )
    return prompt_ids


def metric_timestamp(metrics: Any, *names: str) -> float | None:
    if metrics is None:
        return None
    for name in names:
        value = getattr(metrics, name, None)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def run_once(
    llm: Any,
    sampling_params: Any,
    prompt_ids: Sequence[int],
    run_number: int,
) -> RunResult:
    # Token IDs are passed directly so vLLM executes exactly the verified prompt.
    prompt = {"prompt_token_ids": list(prompt_ids)}
    started = time.perf_counter()
    request_outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
    elapsed = time.perf_counter() - started

    if len(request_outputs) != 1 or len(request_outputs[0].outputs) != 1:
        raise RuntimeError("Expected exactly one request with one completion")

    request_output = request_outputs[0]
    completion = request_output.outputs[0]
    generated_tokens = len(completion.token_ids)
    e2e_tps = generated_tokens / elapsed if elapsed > 0.0 else math.inf

    # RequestMetrics is version-dependent. When available, this matches the C++
    # convention closely: exclude the token produced by prefill and time from
    # first-token completion until request completion. The final EOS token, when
    # present in completion.token_ids, remains included just like qwen_infer.
    metrics = getattr(request_output, "metrics", None)
    first_token_time = metric_timestamp(metrics, "first_token_time")
    finished_time = metric_timestamp(metrics, "finished_time", "finish_time")
    decode_tokens: int | None = None
    decode_seconds: float | None = None
    decode_tps: float | None = None
    if (
        first_token_time is not None
        and finished_time is not None
        and finished_time > first_token_time
    ):
        decode_tokens = max(generated_tokens - 1, 0)
        decode_seconds = finished_time - first_token_time
        decode_tps = decode_tokens / decode_seconds

    return RunResult(
        run=run_number,
        prompt_tokens=len(prompt_ids),
        generated_tokens=generated_tokens,
        elapsed_seconds=elapsed,
        e2e_tps=e2e_tps,
        decode_tokens=decode_tokens,
        decode_seconds=decode_seconds,
        decode_tps=decode_tps,
        text=completion.text,
    )


def print_result(result: RunResult) -> None:
    print(f"\n[Run {result.run}]")
    print(f"prompt_tokens={result.prompt_tokens}")
    print(f"generated_tokens={result.generated_tokens}")
    print(f"e2e_seconds={result.elapsed_seconds:.6f}")
    print(f"e2e_output_tps={result.e2e_tps:.3f}")
    if result.decode_tps is None:
        print("decode_tps=unavailable (this vLLM version exposes no token timestamps)")
    else:
        print(f"decode_tokens={result.decode_tokens}")
        print(f"decode_seconds={result.decode_seconds:.6f}")
        print(f"decode_tps={result.decode_tps:.3f}")
    print("[Generated text]")
    print(result.text)


def configure_hcu_device_count_fallback() -> None:
    """Avoid the known HCU AMDSMI/libdrm ABI crash during device discovery.

    The installed HCU plugin's AMDSMI library requests vendor-specific
    gpu_device_* symbols that Ubuntu's standard libdrm_amdgpu does not export.
    Returning -1 from PyTorch's stateless AMDSMI probe makes vLLM use the
    normal PyTorch/HIP runtime instead. It does not alter model execution.
    """
    if importlib.util.find_spec("vllm_hcu") is None:
        return

    import torch

    if hasattr(torch.cuda, "_device_count_amdsmi"):
        torch.cuda._device_count_amdsmi = lambda: -1
        print("hcu_device_count_backend=pytorch_hip (AMDSMI probe disabled)")


def main() -> None:
    args = parse_args()

    configure_hcu_device_count_fallback()
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "vLLM is not installed in the selected Python environment. "
            "This machine's existing vLLM is installed for /usr/bin/python3; "
            "run this script with that interpreter."
        ) from exc

    print("[Configuration]")
    print(f"model={args.model}")
    print(f"tensor_parallel_size={args.tensor_parallel_size}")
    print(f"max_tokens={args.max_tokens}")
    print(f"user_text={USER_TEXT!r}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        trust_remote_code=args.trust_remote_code,
    )
    tokenizer = llm.get_tokenizer()
    prompt_ids = build_prompt_ids(tokenizer)
    print(f"prompt_ids={prompt_ids}")
    print("prompt_match_qwen_infer=true")

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        ignore_eos=False,
    )

    for warmup in range(args.warmup_runs):
        print(f"[Warmup {warmup + 1}/{args.warmup_runs}]")
        run_once(llm, sampling_params, prompt_ids, 0)

    results = [
        run_once(llm, sampling_params, prompt_ids, run_number)
        for run_number in range(1, args.runs + 1)
    ]
    for result in results:
        print_result(result)

    if len(results) > 1:
        print("\n[Summary]")
        print(f"runs={len(results)}")
        print(
            "mean_e2e_output_tps="
            f"{statistics.fmean(result.e2e_tps for result in results):.3f}"
        )
        decode_rates = [
            result.decode_tps
            for result in results
            if result.decode_tps is not None
        ]
        if len(decode_rates) == len(results):
            print(f"mean_decode_tps={statistics.fmean(decode_rates):.3f}")
        else:
            print("mean_decode_tps=unavailable")


if __name__ == "__main__":
    main()
