# NVIDIA SM100 快速上手与热更新

[English](quickstart.md) | [简体中文](quickstart_zh.md)

本文面向刚拿到仓库、希望在 GPU 物理机上快速拉起 RSInfer、体验**不重启热改配置与代码**、查看进程日志以及运行一键评测的开发者。以 4×SM100 上的 DeepSeek-V4.1-Flash（DS1017，DP4/EP4）为例；通用单机 CLI（含 Qwen3.8、单卡与 CPU 调试）详见 [deploy/README.md](../deploy/README.md)。

在 Ascend 910C 上开发，请从 [部署、调优与验证索引](README.md) 进入。

## 1. 环境准备

激活已安装 PyTorch 与对应算子库（DeepGEMM、FlashMLA、TileKernels、DeepSelect、TileLang）的 Python 环境，并指定模型目录：

```bash
source /path/to/venv/bin/activate
export RSINFER_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

- 也可以不提前激活或导出变量，直接在每条命令里显式传 `--python /path/to/python` 和 `--model /path/to/DeepSeek-V4.1-Flash-1017`。
- `deploy/serve_ds1017.sh` 与 `tests/gpu/ds1017_bench/` 下的脚本会自动查询当前 Python 的 `torch.version.cuda`，将 `CUDA_HOME`、`nvcc`、`LD_LIBRARY_PATH`、`TRITON_PTXAS_PATH` 对齐到对应的 `/usr/local/cuda-<ver>`，并自动把 `nvidia.cu13` 头文件目录加入 `CPATH`（若需要自定义工具链，可传 `--cuda-env <script.sh>`）。

## 2. 一键启动服务与发送请求

```bash
# 后台启动 4 卡 DS1017（默认 GPU 0,1,2,3，端口 8000，运行目录 run/rsinfer）
# 首次加载权重与录制 CUDA Graph 通常需要 2~3 分钟，就绪后自动返回
./deploy/serve_ds1017.sh serve --gpus 0,1,2,3

# 查看引擎进程、HTTP / 控制面地址、当前系统身份和健康状态
./deploy/serve_ds1017.sh status

# 流式对话与补全（自动从 run/rsinfer/engine.json 读取 URL 和模型名）
./deploy/serve_ds1017.sh chat "用三句话介绍一下什么是推理引擎热更新。" --max-tokens 128
./deploy/serve_ds1017.sh complete "The capital of France is" --max-tokens 32
```

也可以直接调用标准 OpenAI 兼容 HTTP API。先用 `/v1/models` 确认实际对外模型名（DS1017 配方默认为 **`DeepSeek-V4.1-Flash`**，启动时可用 `-- --served-model-name <NAME>` 自定义）：

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":32,"temperature":0}'
```

> 常用启动选项：`--foreground`（前台运行直到 Ctrl-C）、`--dry-run`（仅生成拓扑与配置并打印启动命令）、`--port <PORT>`、`--run <RUN_DIR>`、`--mem-fraction <F>`（默认 `0.80`）。完整选项运行 `./deploy/serve_ds1017.sh --help` 查看。

## 3. 在跑着的引擎上玩热变更（`control` / `apply`）

控制面根据变更计划重载受影响的单元。下例修改调度策略，在提交前可预演确认哪些进程和状态会受影响。

### 3.1 查看运行态拓扑、单元与变更日志

```bash
# 控制面状态与所有进程/单元的当前版本
./deploy/serve_ds1017.sh control status

# 查看全部变更历史记录（每次 commit 对应一个新的系统 identity）
./deploy/serve_ds1017.sh control logs

# 导出完整的进程与单元内省信息（JSON 体积较大）
./deploy/serve_ds1017.sh control introspect | less
# 或保存/用 jq 筛选：
./deploy/serve_ds1017.sh control introspect > /tmp/introspect.json
```

> **提示**：`control introspect` 输出的 JSON 较长，直接接 `| head -n 60` 时，`head` 提前关闭管道会导致 Python 客户端在 stderr 打印 `BrokenPipeError`（属于管道提前关闭的正常行为，对前 60 行输出和引擎本身没有任何影响）；建议改用 `| less`、`> /tmp/introspect.json`、`| jq` 或 `2>/dev/null | head -n 60`。

### 3.2 热改配置（不重载权重与 CUDA Graph）

先用 `--dry-run` 预演变更计划，确认动作范围后再真正应用：

```bash
# 预演：查看将修改的配置键与受影响单元
./deploy/serve_ds1017.sh control apply --config '{"sched.overlap": false}' --dry-run

# 热生效
./deploy/serve_ds1017.sh control apply --config '{"sched.overlap": false}' --intent "disable overlap"

# 改回开启
./deploy/serve_ds1017.sh control apply --config '{"sched.overlap": true}' --intent "restore overlap"
```

### 3.3 热改单元代码与查看进程日志（完整闭环 Demo）

从与运行中 identity 对应的源码根目录执行，在临时副本中给 `sched-0` 的策略加探针。保留这份基线源码供最后恢复。 [FCFS 调度策略](../engine/policies/fcfs/unit.py)

**第 1 步：给 `Fcfs.order` 加一行 stderr 日志**

```bash
RSI_BASE=$(pwd -P)
RSI_PROBE=$(mktemp -d)
cp -a "$RSI_BASE/engine" "$RSI_PROBE/engine"
python - "$RSI_PROBE/engine/policies/fcfs/unit.py" <<'PY'
import sys
from pathlib import Path
p = Path(sys.argv[1])
s = p.read_text()
s = s.replace(
    "class Fcfs:\n    def order(self, waiting: Sequence[Candidate]) -> list[int]:\n",
    "import sys\n\n\nclass Fcfs:\n    def order(self, waiting: Sequence[Candidate]) -> list[int]:\n"
    "        if waiting:\n"
    "            print(f'[hot-reload-demo] ordering {len(waiting)} req(s)', file=sys.stderr, flush=True)  # PROBE\n",
)
assert s != p.read_text(), "Fcfs.order did not match the expected source"
p.write_text(s)
PY
```

**第 2 步：静态检查 → 预演计划 → 热应用**

```bash
# 静态检查工作区源码树与单元清单
./deploy/serve_ds1017.sh control check --source "$RSI_PROBE"

# 预演变更计划（可以看到只重载 sched-0 上的 policies.fcfs）
./deploy/serve_ds1017.sh control apply --source "$RSI_PROBE" --dry-run

# 真正热应用到运行中的引擎
./deploy/serve_ds1017.sh control apply --source "$RSI_PROBE" --wait-empty 120 --intent "demo fcfs hot reload"
```

**第 3 步：发请求并查看进程日志确认新代码已生效**

各引擎进程的 stdout/stderr 日志位于 **`<run>/host-a/snapshots/.run/<role>-<index>-<seq>/log`**（例如 `frontend-0-0/log`、`sched-0-1/log`、`worker-0-2/log` ～ `worker-3-5/log`）：

```bash
./deploy/serve_ds1017.sh chat "Hello!" --max-tokens 16
grep "hot-reload-demo" run/rsinfer/host-a/snapshots/.run/sched-0-*/log | tail -n 5
```

看到 `[hot-reload-demo] ordering 1 req(s)`，说明热更新已经在运行中的调度器上生效。

**第 4 步　重新应用基线源码并复验请求**

```bash
./deploy/serve_ds1017.sh control check --source "$RSI_BASE"
./deploy/serve_ds1017.sh control apply --source "$RSI_BASE" --dry-run
./deploy/serve_ds1017.sh control apply --source "$RSI_BASE" --wait-empty 120 --intent "revert demo"
./deploy/serve_ds1017.sh chat "Hello!" --max-tokens 16
```

### 3.4 在途请求处理与框架边界

- **引擎正有在途请求时**：除纯 live 配置键外，默认 `apply` 会被安全拒绝。可按需要加上：
  - `--wait-empty [SECONDS]`：前端暂停接收新请求，等当前在途请求跑完后自动热更；
  - `--force`：立即中止在途请求并执行热更。
- **框架本体边界**：[engine/framework.toml](../engine/framework.toml) 列出的框架核心模块不支持热重载，修改后 `apply` 会以 `escalate.restart_engine` 明确拒绝并要求重启引擎；其余业务单元（调度器、策略、前端、模型、层、kernel、状态等）均支持热改。更详细的热改开发约定与排障见 [.claude/skills/rsinfer/SKILL.md](../.claude/skills/rsinfer/SKILL.md)。

## 4. 释放/恢复显存（`sleep` / `wake`）与停服

临时要把卡让给测试或其他任务、又不想销毁引擎控制面时，可以让引擎进入 `sleep`：

```bash
# 等在途请求结束后释放 4 卡权重与状态池显存（引擎返回 400 拒绝新请求）
./deploy/serve_ds1017.sh sleep --wait-empty
nvidia-smi

# 从 checkpoint 重新加载权重、重建状态池与 CUDA Graph 并恢复服务
./deploy/serve_ds1017.sh wake

# 彻底停止引擎并清理所有进程
./deploy/serve_ds1017.sh stop
```

## 5. 一键运行 DS1017 / AgentX 基准评测

跑评测前先 `stop` 释放 `0,1,2,3`，或直接指定空闲的 4 张卡（例如 `--gpus 4,5,6,7`）。评测脚本会自动把探针打在 `.runs/bench-*` 临时隔离副本中、退出时自动清理，不会污染 git 工作树：

```bash
# 官方 Prefill 单点（64K token budget；不传数字则跑 4K–64K 全档）
./tests/gpu/ds1017_bench/run_rsinfer_official.sh prefill --gpus 0,1,2,3 65536

# 官方 Decode 单点（D=256/GPU；不传数字则跑 4–256 全档）
./tests/gpu/ds1017_bench/run_rsinfer_official.sh decode --gpus 0,1,2,3 256

# AgentX 回放评测（<SUITE_PREFIX> 对应 <prefix>.json 与 <prefix>.i32）
./tests/gpu/ds1017_bench/run_agentx.sh run-rsinfer --gpus 0,1,2,3 /path/to/agentx_eval 32
```

- 更多评测选项、输出格式与数据准备方式见 [tests/gpu/ds1017_bench/README.md](../tests/gpu/ds1017_bench/README_zh.md)。
- 仓库中的主要演进分支也可直接用同一套命令体验：
  - `master`：主干基线；
  - [`optimized-deepinfer`](https://github.com/open-agent-infra/RSInfer/blob/87567b2e2960b3017bbcb5f5cbed9b9778a318ee/tests/gpu/ds1017_bench/README_zh.md)：集成 AKA 优化算子（评测脚本额外支持 `--aka-profile auto|official|validated|expanded` 与 `--aka-warmup`）；
  - [`exp/agentx-opt`](https://github.com/open-agent-infra/RSInfer/blob/9f321de350336549fb594ee127937c3cc443c796/tests/gpu/ds1017_bench/README_zh.md)：面向 AgentX 并发负载的调度与路径优化分支。
