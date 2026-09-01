# HybridAI Benchmark

当前 benchmark 默认覆盖 batch=1、单线程、greedy 生成。

## 构建

```bash
cmake -S . -B build-benchmark \
  -DHYBRIDAI_ENABLE_HIP=ON \
  -DHYBRIDAI_BUILD_BENCHMARKS=ON \
  -DHYBRIDAI_BUILD_OPERATOR_BENCHMARKS=ON
cmake --build build-benchmark --parallel
```

## 延迟

```bash
python3 benchmark/python/benchmark_latency.py \
  --runner build-benchmark/bin/hybridai_benchmark_runner \
  --model /path/to/model \
  --backend hip \
  --max-new-tokens 16 \
  --warmup 1 \
  --runs 5 \
  --output results/latency.jsonl
```

输出包含原始 JSONL 和对应的 `.summary.json`。终端汇总会显示 prompt token
数、实际生成 token 数和 `max_new_tokens`；其中 `generated_tokens` 是模型
实际生成的数量，遇到 EOS 时可能小于 `max_new_tokens`。同时包括 E2E、
TTFT、decode 和 TPS 的均值及 P50/P90/P95/P99。终端表格中的时间单位为
毫秒（`ms`）；JSONL 和 summary JSON 保留 runner 的秒（`s`）原始值。

## 吞吐

```bash
python3 benchmark/python/benchmark_throughput.py \
  --runner build-benchmark/bin/hybridai_benchmark_runner \
  --model /path/to/model \
  --backend hip \
  --max-new-tokens 64 \
  --warmup 1 \
  --runs 5 \
  --output results/throughput.jsonl
```

当前 `throughput` 是串行吞吐。并发、batch 和服务端 QPS 需要 runtime 后续提供对应能力。

## 尚未支持的模块

`benchmark_serving.py`、`benchmark_prefix_caching.py` 和 `benchmark_moe.py` 当前会输出结构化 `skipped` 结果，分别对应 serving API、session/prefix cache、MoE routing telemetry 尚未具备的情况。
