# DS1017 and AgentX benchmark guide

[English](README.md) | [简体中文](README_zh.md)

Run these benchmarks from the repository root on a host with four SM100 GPUs (DP4/EP4). The suite covers:

1. The official DS1017 prefill and decode sweeps on RSInfer and stock DeepInfer.
2. AgentX dataset preparation and replay on both engines.

Measurement probes (`admit_probe.py`, `dsbench_step_probe.py`, `decode_counts_probe.py`, and `top5-measurement.patch`) run in a staged copy under `.runs/`. Successful default runs clean up the staged sources.

---

## 1. Set up the environment

- Set `--gpus 0,1,2,3` or `GPUS` to four available GPUs with less than about 1.5 GiB in use on each.
- Activate a Python 3.12 environment with PyTorch and the DeepInfer kernel dependencies, or select one with `--python` or `RSINFER_PY`. The scripts add the environment's `nvidia.cu13` headers to `CPATH` and look for a CUDA toolkit when needed. Use `--cuda-env <script.sh>` for a custom toolchain.
- Pass the checkpoint directory with `--model /path/to/DeepSeek-V4.1-Flash-1017` or set `RSINFER_MODEL`.

```bash
BENCH=tests/gpu/ds1017_bench
export RSINFER_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

---

## 2. Official DS1017 prefill and decode sweeps

The 131,072-token workload has a 126,976-token cached prefix and a 4,096-token uncached suffix. The benchmark uses YoCo with 140 retained tokens, sampler top-k 1,024, 32 redundant experts, and random expert routing on four GPUs (DP4/EP4).

### Run the RSInfer baseline

```bash
# Measure 64K prefill. Omit the budget to sweep 4096, 8192, 16384, 32768, and 65536.
"$BENCH/run_rsinfer_official.sh" prefill --model "$RSINFER_MODEL" --gpus 0,1,2,3 65536

# Measure decode at D=256/GPU. Omit D to sweep 4, 8, 16, 32, 64, 128, and 256.
"$BENCH/run_rsinfer_official.sh" decode  --model "$RSINFER_MODEL" --gpus 0,1,2,3 256
```

Use `--out <DIR>` to override the default `.runs/results/rsinfer-{prefill,decode}-<timestamp>-<pid>/`. The result files include:
- Prefill: `rsinfer-prefill-<BUDGET>.jsonl` records `prefill_tokens_per_s_per_gpu`, `prefill_window_steps`, and `prefill_window_seconds`.
- Decode: `rsinfer-decode-<D>.json` records `decode_tokens_per_s_per_gpu`, `decode_avg_sps`, and `decode_max_step_ms`.

Common options include `--python <PY>`, `--port <PORT>`, `--out <DIR>`, `--keep-engine` to leave the engine running, and `--run <RUN_DIR>` to reuse an existing engine.

### Run stock DeepInfer

```bash
# Prefill defaults to --mode resident across budgets. Use --mode cold to restart at each budget.
"$BENCH/run_di_official.sh" prefill --model "$RSINFER_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 65536

# Decode runs D<=248 together in one resident engine and starts a fresh engine sized for each D>248.
"$BENCH/run_di_official.sh" decode  --model "$RSINFER_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 256
```

---

## 3. Prepare and replay AgentX

AgentX reconstructs multi-turn and subagent sessions from `semianalysisai/cc-traces-weka-062126-256k`. It rebuilds shared prefixes from real code using 64-token block hashes. Replay uses real expert routing, prefix caching, and YoCo `compact/140`. Use `--yoco-clip 0` to disable YoCo clipping.

### Prepare the dataset

From the Weka JSONL traces, generate `corpus_ids.json`, `agentx_dev.{json,i32}` (16 sessions), and `agentx_eval.{json,i32}` (48 sessions).

```bash
"$BENCH/run_agentx.sh" prepare \
  --model "$RSINFER_MODEL" \
  /path/to/cc-traces-weka-062126-256k.jsonl \
  ./agentx-data-131k
```

Use `agentx_driver.py prepare` directly to customize `--max-context`, `--dev-sessions`, `--eval-sessions`, or `--seed`.

### Replay the benchmark

The suite prefix identifies a pair of files, `<prefix>.json` and `<prefix>.i32`. For example:

```bash
# Replay RSInfer at C=32. The default sweep is 16, 24, 48. Repeat 32 32 to compare cold and warm runs.
"$BENCH/run_agentx.sh" run-rsinfer --model "$RSINFER_MODEL" --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32

# Stock DeepInfer replay
"$BENCH/run_agentx.sh" run-di --model "$RSINFER_MODEL" --di-root /path/to/DeepInfer --gpus 0,1,2,3 ./agentx-data-131k/agentx_eval 32
```

The default output directory is `.runs/results/agentx-{rsinfer,di}-<timestamp>-<pid>/`. Each concurrency level reports:
- `total_tok_s_per_gpu` and `output_tok_s_per_gpu` for total and generated tokens per second per GPU
- `decode_tps_per_user` (p50, p10, mean) for per-user generation speed
- `ttft_s` (p50, p90, p99), `tpot_ms` (p50), and `gap_ms` (p99, max)
- `cached_input_tokens`, `requests_completed`, and `requests_total`

---

## 4. Script index

| File | Purpose |
|---|---|
| `run_rsinfer_official.sh` | Official DS1017 prefill and decode benchmarks on RSInfer |
| `run_di_official.sh` | Official DS1017 prefill and decode benchmarks on stock DeepInfer |
| `run_agentx.sh` | AgentX dataset preparation and replay on either engine |
| `dsbench_pareto_sweep.py` | RSInfer prefill driver using StepChannel and DSBENCH_STEP measurements |
| `dsbench_decode.py` | RSInfer decode driver verifying AGENTX_ADMIT prefix hits and measuring 60 steady-state steps |
| `di_official_sweep.py` | Multi-budget resident prefill and decode driver for stock DeepInfer |
| `agentx_driver.py` | AgentX dataset preparation (prepare) and replay (rsinfer-direct or deepinfer) |
| `agentx_corpus.py` / `agentx_prepare_bank.py` | Extract the code corpus and assemble prefixes from 64-token block hashes |
| `di_prefix_directory.py` / `di_prefix_pages.cpp` / `build_di_prefix_ext.py` | C++ prefix page directory extension for AgentX replay on stock DeepInfer |
| `admit_probe.py` / `dsbench_step_probe.py` / `decode_counts_probe.py` / `top5-measurement.patch` | Timing and prefix-hit instrumentation for the staged benchmark tree |
