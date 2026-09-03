# HybridAI Benchmark

当前 benchmark 默认覆盖 batch=1、单线程、greedy 生成。
MTP 默认关闭；使用 `--enable-mtp` 可开启 Qwen 原生 MTP agreement probing。

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

启用 MTP agreement 统计：

```bash
python3 benchmark/python/benchmark_latency.py \
  --runner build-benchmark/bin/hybridai_benchmark_runner \
  --model /path/to/model \
  --backend hip \
  --max-new-tokens 16 \
  --warmup 1 \
  --runs 5 \
  --enable-mtp \
  --output results/latency-mtp.jsonl
```

启用当前已实现的多 token greedy speculative MTP 路径：

```bash
python3 benchmark/python/benchmark_latency.py \
  --runner build-benchmark/bin/hybridai_benchmark_runner \
  --model /path/to/model \
  --backend hip \
  --max-new-tokens 16 \
  --warmup 1 \
  --runs 5 \
  --speculative-mtp \
  --output results/latency-speculative.jsonl
```

该模式要求同时启用 Qwen 原生 MTP 权重，默认每轮最多生成 8 个 candidate，
支持通过 `--speculative-width 1..16` 调整 proposal width，支持单序列和 greedy decoding。target 会一次 forward 验证整个 candidate 序列，
只提交最长通过前缀；发生 mismatch 时会重新 replay 已接受 token 和 target
correction，保证 Attention/DeltaNet cache 不提交未验证状态。runner 的 JSONL
记录包含 `token_ids`，可用于与 disabled baseline 做逐 token 一致性比较。

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

`--enable-mtp` 仅执行 proposal 与 target greedy token 的 agreement probe，
不会提交多 token proposal，因此可能增加耗时，不能视为 MTP 加速结果。
`--speculative-mtp` 才会执行多 token proposal、target verification 和最长前缀
提交。两种模式的 `mtp_acceptance_rate` 都按实际接受 token 数除以 proposed token
数计算。

### 已验证结果（Qwen3.8-27B，单卡 `gfx936`）

使用 `HIP_VISIBLE_DEVICES=7`、prompt token 数为 23、`max_new_tokens=160`、
预热 1 次、测量 5 次时，模型实际因 EOS 生成 40 个 token：

| 模式 | E2E 均值 | TTFT 均值 | Decode 均值 | Decode TPS | MTP agreement |
|---|---:|---:|---:|---:|---:|
| MTP disabled | 3064.852 ms | 184.433 ms | 2880.270 ms | 13.5405 | — |
| `--enable-mtp` | 3225.784 ms | 191.349 ms | 3034.280 ms | 12.8532 | 80% |

该结果表明当前 agreement probe 比 disabled 基线慢约 5.35%；这是预期的，
因为 probe 额外执行 MTP forward，但候选 token 尚未用于减少 target forward。
真实多 token speculative 路径已接入该 benchmark runner，但仍处于正确性验证阶段。

后续复测（`max_new_tokens=40`、预热 1 次、测量 3 次）得到：baseline E2E
均值为 `3072.153 ms`、decode TPS 为 `13.5071`；MTP agreement probe E2E
均值为 `3207.503 ms`、decode TPS 为 `12.9190`，agreement rate 为 `80%`。
因此当前 `--enable-mtp` 的 E2E 开销约为 `4.40%`，decode TPS 下降约 `4.35%`。
该开关当前应理解为正确性/一致性观测开关，不是性能优化开关。

### 多 token speculative 初步结果

在同一 Qwen3.8-27B、`gfx936`、`max_new_tokens=40`、预热 1 次、测量 3 次的
测试中：

| 模式 | E2E 均值 | Decode TPS | Proposed | Accepted | Fallback |
|---|---:|---:|---:|---:|---:|
| MTP disabled | 3072.153 ms | 13.5071 | 0 | 0 | 0 |
| `--speculative-mtp` | 3316.656 ms | 12.0608 | 37 | 2 | 35 |

当前 acceptance 统计按每轮 proposal 中实际通过的 token 计数，仍需进一步扩大
prompt 和 runs 样本后再下性能结论。即使 acceptance 较高，cache clone、MTP
递归 proposal 和 mismatch replay 也可能抵消批量 verification 的收益。

已完成一组真实模型 smoke：`max_new_tokens=8` 时 speculative 与 baseline 的
完整 `token_ids` 序列一致；speculative 统计为 proposed `6`、accepted `1`、
fallback `1`。该 smoke 只验证功能，不代表稳定性能结果。

最新真实模型 smoke 使用 prompt `Hello`、`max_new_tokens=16`，模型因 EOS 实际
生成 10 个 token。baseline 与 speculative 的完整 `token_ids` 一致：

| 模式 | E2E | Decode TPS | Proposed | Accepted | Fallback |
|---|---:|---:|---:|---:|---:|
| MTP disabled | 921.022 ms | 13.4079 | 0 | 0 | 0 |
| `--speculative-mtp` | 561.288 ms | 17.8340 | 8 | 7 | 1 |

该单次高 acceptance 样本中 E2E 降低约 39.1%，decode TPS 提升约 33.0%。它说明
真正 speculative 路径在 proposal 命中率足够高时能够减少 target decode forward，
但不能替代多 prompt、多轮正式 benchmark。实现同时修正了 full-acceptance 后的
MTP cache 处理：proposal scratch 不包含完整 prompt 上下文，因此不会跨轮直接
复用；下一轮以已提交的 target hidden state 作为状态真值重新建立 draft cache。

对 mismatch 路径也采用保守策略：accepted prefix 可以提交到 target cache，但
correction 之后的 MTP cache 不能通过简单截断得到。因此发生 mismatch 后切换为
target-only greedy decode，直到本次请求结束。这样牺牲部分后续 speculative 机会，
换取 Attention/DeltaNet 状态不会使用错误的 positional context；后续可通过安全的
checkpoint/truncate API 恢复 mismatch 后的 speculative 能力。

随后在同一 `gfx936` 环境、默认 proposal width=8、prompt token 数为 23、
`max_new_tokens=40`、预热 1 次、测量 3 次下复测 speculative，三次结果稳定：

| 模式 | E2E 均值 | Decode TPS | Proposed | Accepted | Fallback |
|---|---:|---:|---:|---:|---:|
| `--speculative-mtp` | 1935.691 ms | 20.6702 | 34 | 25 | 6 |

本轮随后启动 baseline 时出现 `Failed to allocate tensor buffer`，因此没有形成
同一轮的 baseline 对照样本；不能将上表直接解释为相对 baseline 的加速结果。
GPU 显存监控显示各设备仍约 30% 使用率，问题更可能与连续大模型加载的分配碎片或
运行时临时 buffer 峰值有关，后续应通过单进程多模式复用已加载模型，避免反复加载
27B checkpoint。

最新代码重建后再次运行 `Hello`、`max_new_tokens=16` 的 speculative smoke，实际
生成 10 个 token，统计为 proposed `8`、accepted `7`、fallback `1`，且完整
`token_ids` 与 baseline 一致。

随后在 GPU 7 上使用同一 prompt、proposal width=8、`max_new_tokens=16`、预热 1 次、测量 5 次进行
多轮回归。baseline 的 E2E 均值为 `770.688 ms`、decode TPS 为 `13.525`；
speculative 的 E2E 均值为 `463.527 ms`、decode TPS 为 `21.591`。两种模式的
完整 `token_ids` 均一致，speculative 每次 acceptance rate 均为 `87.5%`。
该结果对应当前 `Hello` 高命中率短序列场景，不能替代多 prompt、长序列测试。

启用 `HYBRIDAI_DELTANET_PROFILE=1` 后，GPU 7 上 192 次 DeltaNet 调用的累计
projection profiling 结果为：`in_proj_qkv=299.238 ms`、`in_proj_z=31.776 ms`、
`in_proj_a=13.450 ms`、`in_proj_b=12.979 ms`、`linear_out_proj=32.002 ms`。
同期 grouped convolution 为 `15.919 ms`，recurrent 为 `130.667 ms`。
因此当前最值得研究的是 `in_proj_qkv` 与 recurrent；grouped convolution
不是当前首要瓶颈。projection profiling 只在环境变量开启时同步计时，默认路径
不增加这些同步。

后续算子优化应先评估将 DeltaNet 多个输入 projection 合并为一次 GEMM 的可行性，
同时保留逐 token 数值等价测试；在确认权重布局、输出切分和显存峰值后再实施，
避免因创建拼接权重而抵消 GEMM 合并收益。

### Projection workspace 复用回归（GPU 7）

新增 `Linear::forward_into()` 后，DeltaNet 的 `in_proj_qkv`、`in_proj_z`、
`in_proj_a`、`in_proj_b` 临时输出 buffer 按 device、memory type 和容量复用。
使用 `Hello`、`max_new_tokens=32`、预热 2 次、测量 20 次进行回归：

| 模式 | E2E mean | E2E P50 | E2E P95 | Decode mean | Decode TPS |
|---|---:|---:|---:|---:|---:|
| baseline | 769.939 ms | 770.148 ms | 771.103 ms | 664.740 ms | 13.539 |
| speculative + workspace reuse | 467.411 ms | 467.340 ms | 468.845 ms | 466.969 ms | 21.415 |

speculative 的 proposed/accepted/fallback 分别为 `8/7/1`，acceptance rate 为
`87.5%`，20 次结果一致。与此前 5 次 speculative 均值 `463.527 ms` 相比，
本轮均值为 `467.411 ms`，短序列下仍处于运行波动范围，不能据此宣称 workspace
复用已经带来确定性收益。当前优化的主要价值是降低 allocator 压力和显存碎片风险；
是否减少实际端到端延迟，需要在更长生成长度和多 prompt 场景中继续验证。

### 长序列 GPU 7 对照结果

在 GPU 7 上使用相同 prompt（27 个 prompt token）、proposal width=8、`max_new_tokens=160`、预热 1
次、测量 5 次进行对照。该测试未提前遇到 EOS，5 次均生成完整 160 个 token：

| 模式 | E2E mean | E2E P50 | E2E P95 | Decode mean | Decode TPS |
|---|---:|---:|---:|---:|---:|
| baseline | 12147.950 ms | 12153.717 ms | 12163.976 ms | 11982.895 ms | 13.269 |
| speculative + workspace reuse | 11800.438 ms | 11809.849 ms | 11811.929 ms | 11799.997 ms | 13.559 |

speculative 每次 proposed/accepted/fallback 为 `8/6/1`，acceptance rate 为
`75%`，共 2 个 speculative rounds。相对 baseline，E2E 平均降低约 `2.86%`，
decode TPS 提升约 `2.19%`；该收益明显低于 `Hello` 短序列高 acceptance 场景，
说明当前 mismatch 后进入 target-only fallback 会限制长序列收益。该结果仍是单个
prompt 的 5 次样本，不能代表多 prompt 的最终结论。

### 多 prompt GPU 7 对照结果

在 GPU 7 上使用两个不同 prompt、`max_new_tokens=80`、预热 1 次、测量 3 次进行
对照。两组测试均完成 80 个 token：

| Prompt | 模式 | E2E mean | Decode mean | Decode TPS | Proposed | Accepted | Acceptance |
|---|---|---:|---:|---:|---:|---:|---:|
| GPU kernel fusion | baseline | 6047.704 ms | 5887.177 ms | 13.419 | 0 | 0 | — |
| GPU kernel fusion | speculative | 6098.167 ms | 6097.899 ms | 13.119 | 4 | 0 | 0% |
| Runtime memory allocation | baseline | 6058.280 ms | 5897.410 ms | 13.396 | 0 | 0 | — |
| Runtime memory allocation | speculative | 5953.818 ms | 5953.558 ms | 13.437 | 4 | 2 | 50% |

第一个 prompt 的 acceptance 为 `0%`，speculative E2E 比 baseline 增加约 `0.83%`，
Decode TPS 下降约 `2.24%`；第二个 prompt 的 acceptance 为 `50%`，speculative
E2E 降低约 `1.72%`，Decode TPS 提升约 `0.31%`。结果表明 speculative 收益对
prompt 内容高度敏感：低 acceptance 时 proposal 和首次 verification 成本会抵消
收益。当前实现每次请求最多完成一个 speculative round，随后 mismatch 会进入
target-only fallback，因此需要在安全 cache checkpoint/truncate API 完成后，继续
评估 mismatch 后恢复 speculative 的方案。

## 尚未支持的模块

`benchmark_serving.py`、`benchmark_prefix_caching.py` 和 `benchmark_moe.py` 当前会输出结构化 `skipped` 结果，分别对应 serving API、session/prefix cache、MoE routing telemetry 尚未具备的情况。
