# DS1017 Benchmark 与 AgentX 数据集脚本使用说明

本目录包含可直接通过 `remote-exec` 在 remote GPU 节点（默认 `n1` / `gpu-node-n1`，4×SM100）运行的脚本，覆盖：

1. **DS1017 官方 Prefill / Decode Benchmark**（`agxinfer` 与原版 `DeepInfer` 对照）
2. **AgentX 数据集生成与回放 Benchmark**（`agxinfer` 与原版 `DeepInfer` 对照）

---

## 0. 环境前提与通用配置

在开发机（macOS / Linux）上需要可用的 `remote-exec` 与 `artifact-cli`（用于通过共享 OSS 挂载盘打包同步源码和取回结果）。

先检查目标节点 GPU 是否空闲（建议确认 4 张目标卡显存均 < 1.5 GiB 且无残留 tmux）：

```bash
BENCH=tests/gpu/ds1017_bench
"$BENCH/remote_exec.sh" 'hostname; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader; tmux ls || true'
```

所有 `remote-*` 命令均通过环境变量覆写默认目标和参数：

| 变量 | 默认值 | 说明 |
|---|---|---|
| `TARGET_POD` | `gpu-node-n1` | 目标 Pod 名（`n1`；例如 `n6` 为 `gpu-node-n6`） |
| `CLUSTER_ID` | `<cluster-id>` | remote cluster名 |
| `CLUSTER_NAMESPACE` | `<namespace>` | Namespace |
| `TARGET_CONTAINER` | `pytorch` | 容器名 |
| `REMOTE_SLEEP` | `6` | 串行 `remote-exec` 调用前等待秒数（防 60 秒滚动限流；设为 `0` 跳过） |
| `REMOTE_EXEC_WRAPPER` | `0` | 若本机需要走 `<remote-cli>` 鉴权，设为 `1` |
| `N1_GPUS` | `0,1,2,3` | 使用的 4 张卡编号（DP4/EP4） |
| `OSS_PREFIX` | `s3://<artifact-bucket>/agxinfer/runs` | 开发机 `artifact-cli` 上传/下载前缀 |
| `REMOTE_OSS_DIR` | `/mnt/artifacts/runs` | Pod 内对应挂载目录 |
| `REMOTE_BASE` | `/workspace/ds1017-bench` | Pod 内工作根目录 |
| `AGX_MODEL` | `/models/DeepSeek-V4.1-Flash-1017` | 共享 Lustre 模型权重目录 |
| `AGX_PY` | `/workspace/claude-yoco/venv-opt2/bin/python` | `agxinfer` SM100 环境解释器 |
| `DI_ROOT` | `/workspace/agxinfer-env/DeepInfer` | 原版 DeepInfer 源码目录 |
| `DI_PY` | `/workspace/agxinfer-env/venv-deepinfer/bin/python` | 原版 DeepInfer 解释器 |

> 所有临时探针（`admit_probe.py`、`dsbench_step_probe.py`、`decode_counts_probe.py`、`top5-measurement.patch`）只打在远端临时 stage 出的副本里，**不会修改本地仓库**。

---

## 1. DS 官方 Prefill & Decode Benchmark

固定工况：131,072 上下文（126,976 共享前缀 + 4,096 未缓存后缀）、`yoco_clip=140`、`sampler_force_topk=1024`、32 个冗余专家、随机路由、4×SM100（DP4/EP4）。

### 1.1 跑 `agxinfer`（通过 `remote-exec`）

```bash
BENCH=tests/gpu/ds1017_bench
TAG=opt3-$(date -u +%Y%m%d-%H%M%S)

# 1) Prefill 五档（默认 4096 8192 16384 32768 65536；也可只传部分档位如 16384 32768 65536）
"$BENCH/run_agx_official.sh" remote-prefill "$TAG" 4096 8192 16384 32768 65536

# 2) Decode 七档（默认 4 8 16 32 64 128 256，对应每卡并发 D）
"$BENCH/run_agx_official.sh" remote-decode "${TAG}-dec" 4 8 16 32 64 128 256

# 查看进度 / 退出码（exit 0 表示全部跑完）
"$BENCH/run_agx_official.sh" remote-status "$TAG"

# 强停任务与引擎
"$BENCH/run_agx_official.sh" remote-stop "$TAG"
```

取回结果：日志中每跑完一档会打印 `RESULT s3://<artifact-bucket>/...`，用 `artifact-cli cp` 下载即可：
- Prefill 结果：`agx-prefill-<TAG>-<BUDGET>.jsonl`（关键字段：`prefill_tokens_per_s_per_gpu`、`prefill_window_steps`、`prefill_window_seconds`）
- Decode 结果：`agx-decode-<TAG>-d<D>.json`（关键字段：`decode_tokens_per_s_per_gpu`、`decode_avg_sps`、`decode_max_step_ms`）

### 1.2 跑原版 `DeepInfer`（通过 `remote-exec`）

```bash
BENCH=tests/gpu/ds1017_bench
TAG=di-$(date -u +%Y%m%d-%H%M%S)

# 1) Prefill（默认 DI_MODE=resident 单次加载跑完所有档；设 DI_MODE=cold 则逐档按上游原生冷启）
"$BENCH/run_di_official.sh" remote-prefill "$TAG" 4096 8192 16384 32768 65536

# 若需逐档冷启或指定 Engram QP 数（默认空=上游默认）：
DI_MODE=cold "$BENCH/run_di_official.sh" remote-prefill "${TAG}-cold" 16384 32768
ENGRAM_QPS=32 "$BENCH/run_di_official.sh" remote-prefill "${TAG}-qp32" 16384 32768

# 2) Decode（自动把 D<=248 合并为一次常驻加载，D>248 如 256 单独按精确容量冷启）
"$BENCH/run_di_official.sh" remote-decode "${TAG}-dec" 4 8 16 32 64 128 256

# 查看进度 / 停止
"$BENCH/run_di_official.sh" remote-status "$TAG"
"$BENCH/run_di_official.sh" remote-stop "$TAG"
```

结果同样自动同步到 `${OSS_PREFIX}/di-prefill-<TAG>-di-prefill-<BUDGET>.jsonl` 和 `${OSS_PREFIX}/di-decode-<TAG>-di-decode-<D>.jsonl`。

---

## 2. AgentX 数据集生成与回放 Benchmark

固定工况：基于 `semianalysisai/cc-traces-weka-062126-256k` 构造多轮 + 子 agent 会话结构，使用真实代码语料按 64-token 块哈希重构共享前缀，真实专家路由（`force_random=false`），开启前缀缓存与 YoCo `compact/140`（`YOCO_CLIP=0` 可关闭）。

### 2.1 生成 AgentX 数据集

**方式 A：从原始 Weka traces JSONL 一步生成（推荐）**

直接在远端（或本地有 tokenizer 环境的机器）生成 `corpus_ids.json`、`agentx_dev.{json,i32}` 和 `agentx_eval.{json,i32}`：

```bash
BENCH=tests/gpu/ds1017_bench
TAG=ax-prep-$(date -u +%Y%m%d-%H%M%S)

# 通过 remote-exec 在 n1 上生成（需先将 Weka 原始 JSONL 放到 Pod 可读路径，例如 /shared-fs/... 或 /mnt/artifacts/...）
"$BENCH/run_agentx.sh" remote-prepare "$TAG" \
  /path/on/n1/cc-traces-weka-062126-256k.jsonl \
  /workspace/agentx-data-131k
```

也可直接调用 `agentx_driver.py prepare`（支持自定义上下文上限、会话数和随机种子）：

```bash
python tests/gpu/ds1017_bench/agentx_driver.py prepare \
  --traces-jsonl /path/to/cc-traces-weka-062126-256k.jsonl \
  --tokenizer /models/DeepSeek-V4.1-Flash-1017 \
  --out-dir /workspace/agentx-data-131k \
  --max-context 131072 --dev-sessions 16 --eval-sessions 48 --seed 42
```

生成产物：
- `corpus_ids.json`：真实 Python 代码语料 token 序列
- `agentx_dev.json` + `agentx_dev.i32`：16 个分层采样会话及其共享前缀二进制 token 库
- `agentx_eval.json` + `agentx_eval.i32`：48 个分层采样会话及其共享前缀二进制 token 库

**方式 B：已有独立 `(suite.json, hashes.json.gz, corpus_ids.json)` 时重新物化 `.i32` 输入库**

```bash
python tests/gpu/ds1017_bench/agentx_prepare_bank.py \
  --suite suite.json --hashes hashes.json.gz --corpus corpus_ids.json \
  --out /workspace/agentx-data-131k/eval
```

另附 `agentx_corpus.py`，用于从 Python 标准库按指定 `长度:条数` 生成真实文本/token 对：

```bash
python tests/gpu/ds1017_bench/agentx_corpus.py \
  --tokenizer /models/DeepSeek-V4.1-Flash-1017 \
  --out /tmp/corpus --spec 4000:640
```

### 2.2 跑 AgentX Benchmark（通过 `remote-exec`）

`<REMOTE_SUITE_PREFIX>` 指向不带扩展名的前缀，目录下需同时存在 `<prefix>.json` 和 `<prefix>.i32`（例如 `/workspace/agentx-data-131k/agentx_eval`）。

```bash
BENCH=tests/gpu/ds1017_bench
SUITE=/workspace/agentx-data-131k/agentx_eval
TAG=ax-$(date -u +%Y%m%d-%H%M%S)

# 1) 跑 agxinfer（默认并发 C = 16 24 48，每档前自动热切清缓存）
"$BENCH/run_agentx.sh" remote-agx "$TAG" "$SUITE" 16 24 48

# 2) 跑原版 DeepInfer（自动编译 ds_prefix_pages 扩展，启用原生 fork_seq 前缀缓存与 YoCo=140）
#    注意与 agxinfer 错开使用同一组 4 卡，或用 N1_GPUS=4,5,6,7 分卡跑
"$BENCH/run_agentx.sh" remote-di "${TAG}-di" "$SUITE" 16 24 48

# 查看进度 / 停止
"$BENCH/run_agentx.sh" remote-status "$TAG"
"$BENCH/run_agentx.sh" remote-stop "$TAG"
```

结果文件自动复制到 `${OSS_PREFIX}/agentx-agx-<TAG>-c<C>.json` 与 `${OSS_PREFIX}/agentx-di-<TAG>.json`（包含吞吐、总耗时、TTFT/TPOT 分位数、缓存命中率等汇总字段）。

---

## 3. 节点内直接运行（不经 `remote-exec`）

若已登录到 Pod 内的 stage 目录（如 `/workspace/ds1017-bench/<TAG>`），可直接调用节点内子命令：

```bash
# agxinfer 官方 bench（设置 KEEP_ENGINE=1 可在跑完后保留常驻引擎以便热改复测）
KEEP_ENGINE=1 bash tests/gpu/ds1017_bench/run_agx_official.sh prefill 16384 32768 65536
bash tests/gpu/ds1017_bench/run_agx_official.sh status
bash tests/gpu/ds1017_bench/run_agx_official.sh stop

# DeepInfer 官方 bench
bash tests/gpu/ds1017_bench/run_di_official.sh prefill 16384 32768
bash tests/gpu/ds1017_bench/run_di_official.sh decode 4 8 16 32 64 128 256

# AgentX 数据准备与回放
bash tests/gpu/ds1017_bench/run_agentx.sh prepare /path/to/weka.jsonl /workspace/agentx-data-131k
bash tests/gpu/ds1017_bench/run_agentx.sh run-agx /workspace/agentx-data-131k/agentx_eval 16 24 48
bash tests/gpu/ds1017_bench/run_agentx.sh run-di  /workspace/agentx-data-131k/agentx_eval 16 24 48
```

---

## 4. 目录文件清单

| 文件 | 作用 |
|---|---|
| `remote_exec.sh` | `remote-exec` 远端命令执行封装（默认 `n1`，带限流间隔保护） |
| `run_agx_official.sh` | `agxinfer` 官方 Prefill / Decode bench 入口（本地 `remote-exec` + 节点内执行） |
| `run_di_official.sh` | `DeepInfer` 官方 Prefill / Decode bench 入口（本地 `remote-exec` + 节点内执行） |
| `dsbench_pareto_sweep.py` | `agxinfer` StepChannel Prefill/Decode sweep 驱动 |
| `dsbench_decode.py` | `agxinfer` 131K 共享前缀 Decode sweep 驱动 |
| `di_official_sweep.py` | `DeepInfer` 原生 `examples/benchmark.py` 常驻多点扫点封装 |
| `admit_probe.py` / `dsbench_step_probe.py` / `decode_counts_probe.py` / `top5-measurement.patch` | 仅在远端临时 stage 副本上应用的计时、前缀命中核对与 top-5 logprob 补丁 |
| `run_agentx.sh` | AgentX 数据生成与 `agxinfer` / `DeepInfer` 回放入口（本地 `remote-exec` + 节点内执行） |
| `agentx_driver.py` | AgentX `prepare`（生成套件与 `.i32` 共享前缀库）、`deepinfer`、`agx-direct` 驱动 |
| `agentx_prepare_bank.py` | 从独立 `(suite, hashes, corpus)` 物化 `.i32`/`.json` 输入库 |
| `agentx_corpus.py` | 从 Python 标准库构建真实代码 token 语料 |
| `di_prefix_directory.py` / `di_prefix_pages.cpp` / `build_di_prefix_ext.py` | `DeepInfer` 离线 API 在线前缀缓存目录及 pybind11 页表扩展 |
