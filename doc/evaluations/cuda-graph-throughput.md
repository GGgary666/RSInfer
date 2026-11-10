# CUDA Graph decode 吞吐 benchmark（2026-11-10）

在本次 **SM90-80GB、Qwen3.8-27B 前 4 层随机 BF16 模型、短上下文**测试中，启用 graph 后的稳态 decode 吞吐是原部署 eager 的 **1.61–1.85 倍**；保持 kernel 相同，仅比较 eager 执行与 graph replay，收益是 **1.45–1.84 倍**。这是 worker 侧结果，不是完整 27B 模型或 HTTP 服务吞吐。

## 测量对象与方法

- 主机 `sm90-node-0`，GPU 1：NVIDIA SM90-80GB，SM90，81559 MiB；驱动 580.105.08。
- Python 3.12.7，PyTorch 2.11.0+cu130，CUDA runtime 13.0。CPU torch 线程数为 1，关闭 TF32；没有锁定 GPU 时钟。运行前目标卡无其他计算进程，结束时进程列表只有本次 benchmark。
- 模型是 Qwen3.8-27B 的真实层尺寸、完整 248320 词表，取前 4 层（3 GDN + 1 attention），随机权重 seed=1，checkpoint 7.57 GiB。模型/conv/KV 为 BF16，GDN temporal state 为 FP32。
- 所有配置固定 `batch.max_rows=8`、`batch.max_context_len=256`、KV page size=4、2048 页、8 个 GDN 槽；graph bucket 为 1/2/4/8，预算 1 GiB。
- batch 为 1/4/8，prompt 长度 32；greedy decoding，每次先预热 16 个 decode 步，再测 128 步，计时段位置为 48–175。
- 每个 mode/batch/overlap 配置先完整走过一次相同的长度序列，记录为 `priming`，不进入稳态统计。之后正式测 3 次；轮换 mode 顺序，交替正序/倒序遍历 batch/overlap，共 **54 次正式测量**。
- overlap 关：每步提交后等结果，深度 1；开：最多保留 2 步，输入通过设备上的 `from_previous` 交接。
- 主指标为 wall-clock：`batch × 128 / 秒数`。计时包含 Driver 生成记录、executor 的元数据/输入处理、完整 forward、采样、结果回传和等待；开始前和结束后同步 CUDA。模型加载、capture、prefill、预热不计入。
- 没有 HTTP、tokenizer、真实 scheduler/IPC 开销，也没有跨步 clone logits。只保留 timing events 和 token hash，避免保留 eager logits 改变内存分配行为。

三种 mode：

| 名称 | 平台 / 配置 | decode attention |
|---|---|---|
| 原部署 eager | `torch_eager` + off，对应部署 `--cuda-graph off` | 原逐请求、变长 PyTorch SDPA |
| 同 kernel eager | `cuda_graph` + off | 与 graph 相同的固定容量、批量 SDPA |
| graph | `cuda_graph` + required | 固定容量、批量 SDPA，完整 forward replay |

原部署开关会同时改变 attention 路径，因此不能把它的全部收益归因于 CUDA Graph；第二组对照用于分离 replay 的收益。

## 原部署开关对比

以下均为 3 次正式测量的中位数。吞吐是所有行合计的 output token/s；每步延迟是 wall time / 128，各行每步生成一个 token。倍率为两个吞吐中位数之比。

| Batch | Overlap | 原 eager tok/s | Graph tok/s | 吞吐倍率 | 每步 ms：eager → graph |
|---:|:---:|---:|---:|---:|---:|
| 1 | 关 | 167.3 | 289.2 | 1.73× | 5.977 → 3.458 |
| 1 | 开 | 191.0 | 350.7 | 1.84× | 5.237 → 2.851 |
| 4 | 关 | 607.5 | 1033.6 | 1.70× | 6.584 → 3.870 |
| 4 | 开 | 677.2 | 1254.7 | 1.85× | 5.907 → 3.188 |
| 8 | 关 | 1128.1 | 1812.8 | 1.61× | 7.092 → 4.413 |
| 8 | 开 | 1249.8 | 2184.2 | 1.75× | 6.401 → 3.663 |

原 eager 与 graph 的 token hash 在 batch 1/4 的 12 组配对中相同，在 batch 8 的 6 组配对中不同。这里保持了相同 prompt、固定生成长度、greedy 参数和计算形状，但**不声称原部署开关在 BF16 下逐 token 等价**。两条 attention 路径不同；本次没有测量这 6 组分歧的逐步 logits 误差。

## 相同 kernel：单独看 graph replay

| Batch | Overlap | 同 kernel eager tok/s | Graph tok/s | 吞吐倍率 |
|---:|:---:|---:|---:|---:|
| 1 | 关 | 168.2 | 289.2 | 1.72× |
| 1 | 开 | 190.2 | 350.7 | 1.84× |
| 4 | 关 | 642.3 | 1033.6 | 1.61× |
| 4 | 开 | 720.3 | 1254.7 | 1.74× |
| 8 | 关 | 1253.7 | 1812.8 | 1.45× |
| 8 | 开 | 1402.7 | 2184.2 | 1.56× |

同 kernel 的 **18 组 graph/eager 配对，完整计时段 token hash 全部相同**。所以原部署 batch=8 的分歧并未出现在仅切换 replay 的对照中。

以 overlap 开启为例，executor 的 host submit 中位耗时从同 kernel eager 的 **5.16 / 5.40 / 5.48 ms** 降至 graph 的 **0.75 / 0.78 / 0.84 ms**（batch 1/4/8）。这支持“graph 减少逐算子 host 提交开销”的解释；wall-clock 收益并不等于 host 耗时降幅，因为设备计算仍需完成。

CUDA event 记录的步跨度从 **5.23 / 5.47 / 5.56 ms** 降至 **2.84 / 3.17 / 3.64 ms**。该跨度包含 kernel 间等待，不能解释为纯 GPU kernel 计算时长。

## 波动、首次运行与构建成本

下表是 3 次正式吞吐的最小值–最大值，不是置信区间；没有删除正式测量中的慢样本。

| Batch / Overlap | 原 eager tok/s 范围 | 同 kernel eager tok/s 范围 | Graph tok/s 范围 |
|---|---:|---:|---:|
| 1 / 关 | 167.14–167.99 | 168.15–168.46 | 286.46–290.39 |
| 1 / 开 | 187.83–191.54 | 189.32–190.55 | 350.68–350.79 |
| 4 / 关 | 607.37–609.67 | 640.93–643.30 | 1033.42–1034.66 |
| 4 / 开 | 667.61–678.64 | 718.02–723.00 | 1254.43–1255.03 |
| 8 / 关 | 1033.22–1133.24 | 1250.38–1255.75 | 1807.93–1813.51 |
| 8 / 开 | 1236.80–1253.85 | 1401.61–1412.53 | 2184.16–2184.58 |

最初仅预热 16 步的探索运行中，首个原 eager batch=1/depth=1 样本只有 25.43 tok/s，后续重复约 160 tok/s。因此改为预先遍历完整长度序列，再重新测量全部配置。正式运行的同一首段 priming 仍只有 **26.12 tok/s、4.900 秒**，而随后的正式样本稳定在 167 tok/s 左右。数据证明存在首次路径开销，但没有 profiling 证据将其归因于某个具体库、编译或缓存机制；不能把这个首次低值用于夸大稳态加速。

4 个 bucket 的 warm-up/capture/验证累计耗时为 **0.303 / 0.236 / 0.242 秒**；构建峰值增量首次约 **350 MiB**，复用平台 capture stream 后约 **318 MiB**。这些是构建阶段的数据，不是额外常驻图大小，也没有包含全部加载耗时。短生命周期服务需要另计启动成本。

## 复现与原始记录

脚本：[tests/gpu/bench_cuda_graph.py](../../tests/gpu/bench_cuda_graph.py)。无需额外依赖；复用已配置的 OSS GPU 环境：

```sh
RID=20261110T103945Z_acd5dd0-dirty_2f4240d7
bash oss_run_agxinfer_cuda_graph/scripts/ossrun/run.sh --release "$RID" \
  --inline 'bash oss_run_agxinfer_cuda_graph/scripts/ossrun/gpu.sh exec python -m tests.gpu.bench_cuda_graph --work-dir /checkpoint/binary/agxinfer_cuda_graph/bench-cuda-graph --output /checkpoint/binary/agxinfer_cuda_graph/bench-cuda-graph/result-steady.json' \
  --name graph-throughput-steady --expect '^AGXINFER_GRAPH_BENCH_OK$' \
  --max-wait-s 1800 --log-out .ossrun-artifacts/graph-throughput-steady.log
```

- 引擎代码 commit `acd5dd03e5ff66e462ea59fdad7342730d681894`；release 额外包含 benchmark 脚本，未修改引擎实现。
- 正式 command：`cmd_20261110T104006Z_graph-throughput-steady_dcb40af5`；done / exit 0 / 成功标记匹配。
- 远程实际测量约为 2026-11-10 18:42:45–18:44:08（UTC+8）；OSS 等待调度不在计时范围内。
- 本地完整日志 `.ossrun-artifacts/graph-throughput-steady.log`，解析后的数据 `.ossrun-artifacts/graph-throughput-steady.json`；远程 JSON 为上方 `--output` 路径。日志的 `BENCH_RESULT` 行包含完整环境、逐次结果、priming 和构建数据。
- 探索 command：`cmd_20261110T103250Z_graph-throughput_5e52d032`，日志/JSON 为 `.ossrun-artifacts/graph-throughput.log` / `graph-throughput.json`，保留首次慢样本。生成物不提交 Git。
- 脚本通过 pyright；没有以不稳定的速度阈值新增 pytest 断言。

尚未覆盖完整 64 层/真实权重、长上下文、其他 GPU、真实请求调度和端到端服务负载。上面的倍率只适用于明确列出的测量条件；后续性能优化可用同 kernel 对照作为基线。
