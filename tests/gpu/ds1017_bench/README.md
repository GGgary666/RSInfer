# DS1017 Benchmark 与 AgentX 数据集脚本使用说明

本目录包含可直接登录 GPU 物理机一键运行的评测脚本（4×SM100，DP4/EP4），覆盖：

1. **DS1017 官方 Prefill / Decode Benchmark**（`RSInfer` 与原版 `DeepInfer` 对照）
2. **AgentX 数据集生成与回放 Benchmark**（`RSInfer` 与原版 `DeepInfer` 对照）

> 所有临时测量探针（`admit_probe.py`、`dsbench_step_probe.py`、`decode_counts_probe.py`、`top5-measurement.patch`）都会自动拷贝到 `.runs/` 下的隔离副本里执行并在结束后清理，**不会修改你的 git 工作树**。

---

## 0. 环境准备

- 4 张空闲 SM100 GPU（建议每张显存占用 < 1.5 GiB），可用 `--gpus 0,1,2,3`（或环境变量 `GPUS`）指定卡号。
- 已激活安装好 PyTorch + DeepInfer 算子依赖的 Python 3.12 环境（也可通过 `--python /path/to/venv/bin/python` 或 `RSINFER_PY` 指定）；脚本会自动从该环境的 `nvidia.cu13` 补齐 `CPATH`，若 `nvcc` 不在 `PATH` 中会自动探测 `/usr/local/cuda-13.2/bin` 或 `/usr/local/cuda/bin`（也可用 `--cuda-env <script.sh>` 指定）。
- DS1017 权重目录通过 `--model /path/to/DeepSeek-V4.1-Flash-1017`（或环境变量 `RSINFER_MODEL`）传入。

```bash
BENCH=tests/gpu/ds1017_bench
export RSINFER_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

---

## 1. DS1017 官方 Prefill & Decode Benchmark

固定工况：131,072 上下文（126,976 共享前缀 + 4,096 未缓存后缀）、`yoco_clip=140`、`sampler_force_topk=1024`、32 个冗余专家、随机路由、4×SM100（DP4/EP4）。

### 1.1 跑 `RSInfer`

```bash
# Prefill（不传档位默认跑 4096 8192 16384 32768 65536；也可以只测单点如 65536）
"$BENCH/run_rsinfer_official.sh" prefill --model "$RSINFER_MODEL" --gpus 0,1,2,3 65536

# Decode（不传档位默认跑 4 8 16 32 64 128 256；也可以只测单点如 256）
"$BENCH/run_rsinfer_official.sh" decode  --model "$RSINFER_MODEL" --gpus 0,1,2,3 256
```

结果默认保存在 `.runs/results/rsinfer-{prefill,decode}-<timestamp>-<pid>/`（可用 `--out <DIR>` 指定）：
- Prefill：`rsinfer-prefill-<BUDGET>.jsonl`（关键字段：`prefill_tokens_per_s_per_gpu`、`prefill_window_steps`、`prefill_window_seconds`）
- Decode：`rsinfer-decode-<D>.json`（关键字段：`decode_tokens_per_s_per_gpu`、`decode_avg_sps`、`decode_max_step_ms`）

常用选项：`--python <PY>`、`--port <PORT>`、`--out <DIR>`、`--keep-engine`（跑完保留引擎供复用）、`--run <RUN_DIR>`（复用已启动的引擎）。

### 1.2 跑原版 `DeepInfer` 对照

```bash
# Prefill（默认 --mode resident 单次常驻跑全档；--mode cold 逐档冷启）
"$BENCH/run_di_official.sh" prefill --model "$RSINFER_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 65536

# Decode（D<=248 合并常驻，D>248 自动按精确容量冷启）
"$BENCH/run_di_official.sh" decode  --model "$RSINFER_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 256
```

---

## 2. AgentX 数据集生成与回放 Benchmark

固定工况：基于 `semianalysisai/cc-traces-weka-062126-256k` 构造多轮 + 子 agent 会话结构，使用真实代码语料按 64-token 块哈希重构共享前缀，真实专家路由（`force_random=false`），开启前缀缓存与 YoCo `compact/140`（`--yoco-clip 0` 可关闭）。

### 2.1 生成 AgentX 数据集

从原始 Weka traces JSONL 一步生成 `corpus_ids.json`、`agentx_dev.{json,i32}`（16 会话）和 `agentx_eval.{json,i32}`（48 会话）：

```bash
"$BENCH/run_agentx.sh" prepare \
  --model "$RSINFER_MODEL" \
  /path/to/cc-traces-weka-062126-256k.jsonl \
  ./agentx-data-131k
```

也可直接调用 `agentx_driver.py prepare` 自定义 `--max-context`、`--dev-sessions`、`--eval-sessions`、`--seed`。

### 2.2 回放 Benchmark

`<SUITE_PREFIX>` 指向 `<prefix>.json` 与 `<prefix>.i32`（例如 `./agentx-data-131k/agentx_eval`）：

```bash
# 跑 RSInfer（默认并发 16 24 48；也可传单点如 32，或重复传 32 32 测冷/热态）
"$BENCH/run_agentx.sh" run-rsinfer --model "$RSINFER_MODEL" --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32

# 跑原版 DeepInfer 对照
"$BENCH/run_agentx.sh" run-di --model "$RSINFER_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```

结果默认保存在 `.runs/results/agentx-{rsinfer,di}-<timestamp>-<pid>/`，每档摘要指标：
- `total_tok_s_per_gpu` / `output_tok_s_per_gpu`：单卡总吞吐与生成吞吐
- `decode_tps_per_user`（`p50` / `p10` / `mean`）：每用户生成速度（Interactivity）
- `ttft_s`（`p50` / `p90` / `p99`）、`tpot_ms`（`p50`）、`gap_ms`（`p99` / `max`）
- `cached_input_tokens`、`requests_completed` / `requests_total`

---

## 3. 文件索引

| 文件 | 作用 |
|---|---|
| `run_rsinfer_official.sh` | DS1017 官方 Prefill / Decode 一键本地评测入口（RSInfer） |
| `run_di_official.sh` | DS1017 官方 Prefill / Decode 一键本地评测入口（原版 DeepInfer） |
| `run_agentx.sh` | AgentX 数据生成与回放一键本地评测入口（RSInfer & DeepInfer） |
| `dsbench_pareto_sweep.py` | RSInfer 官方 Prefill 测量驱动（通过 `StepChannel` 驱动并从 `DSBENCH_STEP` 统计） |
| `dsbench_decode.py` | RSInfer 官方 Decode 测量驱动（核验 `AGENTX_ADMIT` 前缀命中并统计 60 步稳态 decode） |
| `di_official_sweep.py` | 原版 DeepInfer 官方 Prefill / Decode 常驻多档驱动 |
| `agentx_driver.py` | AgentX 数据集生成（`prepare`）与回放驱动（`rsinfer-direct` / `deepinfer`） |
| `agentx_corpus.py` / `agentx_prepare_bank.py` | 真实代码语料提取与按 64-token 块哈希拼装前缀辅助模块 |
| `di_prefix_directory.py` / `di_prefix_pages.cpp` / `build_di_prefix_ext.py` | 原版 DeepInfer AgentX 回放用 C++ 前缀页表扩展 |
| `admit_probe.py` / `dsbench_step_probe.py` / `decode_counts_probe.py` / `top5-measurement.patch` | 自动施加于临时隔离副本的计时与前缀命中探针（不修改仓库源码） |
