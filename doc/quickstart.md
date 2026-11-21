# RSInfer quickstart and live updates

[English](quickstart.md) | [简体中文](quickstart_zh.md)

Start DeepSeek-V4.1-Flash (DS1017) on four SM100 GPUs with DP4/EP4, send a request, and change the running scheduler. For the general single-host CLI, including Qwen3.8, single-GPU serving, and CPU debugging, see the [deployment guide](../deploy/README.md) (Chinese).

## 1. Set up the environment

Activate a Python environment with PyTorch, DeepGEMM, FlashMLA, TileKernels, DeepSelect, and TileLang installed. Point the launcher at your checkpoint.

```bash
source /path/to/venv/bin/activate
export RSINFER_MODEL=/path/to/DeepSeek-V4.1-Flash-1017
```

- Alternatively, pass `--python /path/to/python` and `--model /path/to/DeepSeek-V4.1-Flash-1017` explicitly where needed.
- The launcher and benchmark scripts check `torch.version.cuda`, look for a matching CUDA toolkit and `nvcc`, and add the `nvidia.cu13` headers to `CPATH`. To supply your own toolchain environment, pass `--cuda-env <script.sh>`.

## 2. Start serving and send a request

```bash
# Start in the background on GPUs 0–3. Defaults: port 8000, run directory run/rsinfer.
# The command returns when ready. Loading weights and capturing graphs typically takes 2–3 minutes.
./deploy/serve_ds1017.sh serve --gpus 0,1,2,3

# Show the processes, HTTP and control endpoints, system identity, and health.
./deploy/serve_ds1017.sh status

# Stream a chat or completion using the URL and model name in run/rsinfer/engine.json.
./deploy/serve_ds1017.sh chat "Explain live updates in an inference engine in three sentences." --max-tokens 128
./deploy/serve_ds1017.sh complete "The capital of France is" --max-tokens 32
```

You can also use the OpenAI-compatible HTTP API. Check the served name with `/v1/models`. The DS1017 recipe defaults to `DeepSeek-V4.1-Flash`. To override it at startup, pass `-- --served-model-name <NAME>`.

```bash
curl -s http://127.0.0.1:8000/v1/models

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-V4.1-Flash","messages":[{"role":"user","content":"1+1=?"}],"max_tokens":32,"temperature":0}'
```

Use `--foreground` to stay attached until Ctrl-C, or `--dry-run` to print the topology, configuration, and launch command. Other options include `--port <PORT>`, `--run <RUN_DIR>`, and `--mem-fraction <F>` (default `0.80`). See `./deploy/serve_ds1017.sh --help` for the full list.

## 3. Update the running engine

RSInfer manages code and state as units. For each edit the control plane determines which units must reload. The configuration and scheduler changes below preserve the loaded weights and KV/SWA state.

### 3.1 Inspect the topology and change history

```bash
# Show the current process and unit versions.
./deploy/serve_ds1017.sh control status

# Each committed change has its own system identity.
./deploy/serve_ds1017.sh control logs

# Page through the full process and unit inventory.
./deploy/serve_ds1017.sh control introspect | less
# Or save the JSON for later inspection with jq.
./deploy/serve_ds1017.sh control introspect > /tmp/introspect.json
```

The introspection JSON can be large. Piping it to `head` may print `BrokenPipeError` when the reader exits early. Use `less`, redirect to a file, or select fields with `jq`.

### 3.2 Change a live configuration key

Preview the change to `sched.overlap`, apply it, and restore the original setting. This key can change without reloading weights or recapturing CUDA graphs.

```bash
# Inspect the affected keys and units.
./deploy/serve_ds1017.sh control apply --config '{"sched.overlap": false}' --dry-run

# Apply the change.
./deploy/serve_ds1017.sh control apply --config '{"sched.overlap": false}' --intent "disable overlap"

# Restore overlap.
./deploy/serve_ds1017.sh control apply --config '{"sched.overlap": true}' --intent "restore overlap"
```

### 3.3 Reload scheduler code and check its logs

The [FCFS scheduling policy](../engine/policies/fcfs/unit.py) runs in `sched-0`. Reloading it takes seconds. The four GPU workers keep their weights and captured graphs. Start this example with a clean copy of the policy file so the revert in step 4 restores it.

**1. Log a call to `Fcfs.order` on stderr.**

```bash
python - <<'PY'
from pathlib import Path
p = Path("engine/policies/fcfs/unit.py")
s = p.read_text()
s = s.replace(
    "class Fcfs:\n    def order(self, waiting: Sequence[Candidate]) -> list[int]:\n",
    "import sys\n\n\nclass Fcfs:\n    def order(self, waiting: Sequence[Candidate]) -> list[int]:\n"
    "        if waiting:\n"
    "            print(f'[hot-reload-demo] ordering {len(waiting)} req(s)', file=sys.stderr, flush=True)\n",
)
p.write_text(s)
PY
```

**2. Check the source, preview the plan, and apply it.**

```bash
# Check the workspace source and unit manifests.
./deploy/serve_ds1017.sh control check --source .

# The plan should reload policies.fcfs on sched-0.
./deploy/serve_ds1017.sh control apply --source . --dry-run

# Apply the change to the running engine.
./deploy/serve_ds1017.sh control apply --source . --intent "demo fcfs hot reload"
```

**3. Send a request and inspect the scheduler log.**

Each process writes stdout and stderr to `<run>/host-a/snapshots/.run/<role>-<index>-<seq>/log`. Examples include `frontend-0-0/log`, `sched-0-1/log`, and `worker-0-2/log` through `worker-3-5/log`.

```bash
./deploy/serve_ds1017.sh chat "Hello!" --max-tokens 16
grep "hot-reload-demo" run/rsinfer/host-a/snapshots/.run/sched-0-*/log | tail -n 5
```

A line such as `[hot-reload-demo] ordering 1 req(s)` confirms the running scheduler executed the new code.

**4. Restore the policy file and apply it to the running engine.**

```bash
git checkout -- engine/policies/fcfs/unit.py
./deploy/serve_ds1017.sh control apply --source . --intent "revert demo"
```

### 3.4 In-flight requests and the framework boundary

- With requests in flight, the engine normally declines `apply` unless the edit only touches live configuration keys. To proceed:
  - `--wait-empty [SECONDS]` pauses admission and waits for outstanding requests to finish.
  - `--force` aborts outstanding requests and applies the update immediately.
- A change to a framework module listed in [engine/framework.toml](../engine/framework.toml) returns `escalate.restart_engine` and requires a restart. The scheduler, policies, frontend, models, layers, kernels, and state units have their own reload rules. The [RSInfer development skill](../.claude/skills/rsinfer/SKILL.md) (Chinese) covers these rules and troubleshooting.

## 4. Release GPU memory, wake the engine, or stop it

Use `sleep` to free the GPUs for other work while keeping the control plane available.

```bash
# Drain requests and release weights and state pools. New requests receive HTTP 400.
./deploy/serve_ds1017.sh sleep --wait-empty
nvidia-smi

# Reload checkpoint weights, rebuild state pools and graphs, and resume serving.
./deploy/serve_ds1017.sh wake

# Stop the engine and its processes.
./deploy/serve_ds1017.sh stop
```

## 5. Run the DS1017 and AgentX benchmarks

Stop the service to free GPUs 0–3, or select four other available GPUs, for example `--gpus 4,5,6,7`. Benchmark probes run in a staged copy under `.runs/`. Successful default runs clean up the staging directory.

```bash
# One prefill point at 64K tokens/GPU. Omit the budget to sweep 4K–64K.
./tests/gpu/ds1017_bench/run_rsinfer_official.sh prefill --gpus 0,1,2,3 65536

# One decode point at 256 requests/GPU. Omit the concurrency to sweep 4–256.
./tests/gpu/ds1017_bench/run_rsinfer_official.sh decode --gpus 0,1,2,3 256

# The replay suite prefix identifies a .json file and its .i32 token file.
./tests/gpu/ds1017_bench/run_agentx.sh run-rsinfer --gpus 0,1,2,3 /path/to/agentx_eval 32
```

- See the [benchmark guide](../tests/gpu/ds1017_bench/README.md) for options, result fields, and dataset preparation.
- Choose the branch for your workload:
  - `master` runs the baseline.
  - [`optimized-deepinfer`](https://github.com/open-agent-infra/RSInfer/blob/87567b2e2960b3017bbcb5f5cbed9b9778a318ee/tests/gpu/ds1017_bench/README.md) supplies AKA prefill kernels and the `--aka-profile` and `--aka-warmup` options.
  - [`exp/agentx-opt`](https://github.com/open-agent-infra/RSInfer/blob/9f321de350336549fb594ee127937c3cc443c796/tests/gpu/ds1017_bench/README.md) contains scheduling and decode optimizations for concurrent AgentX traffic.
