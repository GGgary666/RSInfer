# Ascend 910C 推理调优与验证

从仓库根目录执行下列命令。启动和停服按模型查部署页，性能对照先固定输入、采样、缓存冷热、拓扑和引擎 identity。 [Ascend 部署](../deploy/ascend.md) · [实验记录模板](../.claude/skills/rsinfer-ascend-optimize/references/experiment-record.md) · [文档导航](README.md)

## 1. 核对设备和运行环境

在启动 worker 的同一 Python / 容器中加载所装 CANN 的 `set_env.sh`，记录 Toolkit、OPP 和驱动版本。用 `-m` 对照物理 die，`-l` 看拓扑，设备表看健康和占用。

```sh
npu-smi info
npu-smi info -m
npu-smi info -l
command -v msprof
python - <<'PY'
import os
import sys
import torch
import torch_npu

print(sys.executable, torch.__version__, torch_npu.__version__, torch_npu.__file__)
for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES"):
    print(key, os.environ.get(key))
PY
```

单 die worker 会在进程内使用 `npu:0`。以实际 worker 的环境和日志核对物理 die、rank、OPP 与扩展。A3 上 `npu-smi` 可能只显示 `Ascend910`，编译自定义算子前可用 `asys info -r=hardware` 核实 SoC。 [环境与运行 skill](../.claude/skills/rsinfer-ascend-runtime/SKILL.md)

## 2. 用实际 worker 采集性能

已有 trace 时先找目标 rank、stream 和请求阶段。需要补采时，在原生实例的独立探针工作区包住图外的执行循环。`steps` 是本次窗口内本来要执行的 `active_steps` 个步骤，`run_step(step)` 代表原调用。分别采预热后的 prefill 和稳定 decode，混合步单独标记。 [引擎内探针](../.claude/skills/rsinfer/SKILL.md#在引擎里做检查排查和验证)

```python
from torch_npu import profiler as p

with p.profile(
    activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
    schedule=p.schedule(wait=0, warmup=0, active=active_steps, repeat=1),
    on_trace_ready=p.tensorboard_trace_handler(str(trace_dir), worker_name=f"rank-{rank}"),
    record_shapes=True,
    experimental_config=p._ExperimentalConfig(
        profiler_level=p.ProfilerLevel.Level1,
        aic_metrics=p.AiCMetrics.PipeUtilization,
    ),
) as prof:
    for step in steps:
        run_step(step)
        prof.step()
```

每次回调只处理一步时，实例持有同一采集器，在窗口首步前 `start()`、每步后 `step()`、结束或异常清理时 `stop()`。在 `ASCEND_PROFILER_OUTPUT` 查 `trace_view.json` 和 `kernel_details.csv`。多 rank 对齐关键路径上的 host 提交、Cube/Vector、HBM/KV 搬运和 HCCL 等待，保留原始 trace、identity、die 与窗口。 [CANN 采集说明](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/model/model-infer-profiling/SKILL.md) · [实测 trace 的解读](rsi/2026-11-26-glm53-native.md#优化后-cann-profile)

独立复现程序可用本机 `msprof`。观察 HCCL 重叠时，`REPRO_SCRIPT` 应启动参与通信的所有 rank。正式服务计时前撤掉探针和同步点。

```sh
msprof --runtime-api=on --task-time=on --hccl=on --aic-metrics=PipeUtilization \
  --output="$TRACE_DIR" python "$REPRO_SCRIPT"
```

## 3. 接入候选与验证正确性

在真实调用点保存实际 shape、stride、dtype、分页布局和有效长度。先比较候选与参考的输出及写回状态，再用同一请求检查是否命中目标实现。热更后核对各 rank 的 `local_pid`、identity、首个不同 token 或 logits。 [算子契约与派发](../.claude/skills/rsinfer-ascend-kernel/SKILL.md) · [精度与运行故障](../.claude/skills/rsinfer-ascend-debug/SKILL.md)

下面的 CPU 回归覆盖 Qwen3 分段 prefill/decode、GLM 图元数据与 refit、热更后的缓存清单处理。ACL Graph 实际回放和 HCCL 集合通信在目标 die 上用相同请求复核。 [已知问题与复验记录](framework-bugs/README.md#ascend-与-cann-案例)

```sh
OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -m pytest -q \
  tests/units/models/test_qwen3.py tests/units/models/test_glm53_graph_metadata.py \
  tests/units/state/test_glm53_cache.py
```

## 4. 端到端对照与采纳记录

### Qwen3 流式基准

从 `status` 取服务的 `QWEN_URL`，使用部署时的 `QWEN_MODEL`。每次将 `BENCH_OUT` 指向一个新目录。两轮保持 tokenizer、salt、并发、长度和缓存状态一致。

```sh
QWEN_URL=http://127.0.0.1:18650
BENCH_OUT="$(mktemp -d)/qwen"
python deploy/bench_ascend.py qwen --url "$QWEN_URL" --tokenizer "$QWEN_MODEL/tokenizer.json" \
  --samples 16 --concurrency 4 --prompt-tokens 512 --output-tokens 64 \
  --repeat 3 --salt 0 --out "$BENCH_OUT"
```

读 `parameters.json` 和 `run-*.json`，按同一 repeat 和请求顺序比较 `output_sha256`（流式文本）、输出 tok/s、TTFT 和 TPOT。逐 token 或 logits 验收需重放固定输入并取对应输出。 [Qwen3 基线与回退案例](rsi/2026-11-26-ascend-a3.md#qwen3-06b)

### GLM AISBench 对照

`ENGINE_RUN` 选择原生或 vLLM 的独立运行目录。`AISBENCH_SUITE` 指向同一外部套件，内含 `prepare.py`、`glm_bench_adapter.py`、`run_benchmark.py` 和 `summarize.py`。`ACCURACY_CHECK` 指向配套精度脚本。原 16K 冷组和 60K 暖组的生成器会把输出目录末级名写入提示词，对照可用 `baseline/glm-8192-32` 与 `candidate/glm-8192-32` 两个新目录，并比对两组测试集、暖机正文和 salt。

```sh
BENCH_OUT="$(mktemp -d)/glm-8192-32"
python deploy/bench_ascend.py glm53 --run "$ENGINE_RUN" --suite "$AISBENCH_SUITE" \
  --accuracy-check "$ACCURACY_CHECK" --out "$BENCH_OUT"
```

查看套件生成的 `summary.csv`、`requests.csv` 和 `accuracy.json`。`experiment.json` 的 `accuracy_exit_code` 为 `0` 表示脚本通过，非零表示失败，`null` 表示未执行。原生的 `launch.identity`、配置和拓扑取自已提交的变更，vLLM 取自成功启动记录。按冷暖组报告成功请求、有效输出 tok/s、TTFT/TPOT 均值与 P90，并注明并行、精度和缓存差异。 [GLM 逐轮结果与采纳依据](rsi/2026-11-26-glm53-native-rsi-iter.md#逐轮结果)

### MemCache 跨重启验收

1. 记录池和推理 PID。原生 `/v1/completions` 接收文本，保存包含 `model`、`prompt`、`cache_salt`、`temperature: 0` 与 `max_tokens` 的请求 JSON。使用固定 tokenizer，令前缀覆盖完整 KV 页，记录编码后的 token IDs、冷请求结果和池写入，然后用 `cache-save` 导出清单。 [原生保存与恢复命令](../deploy/ascend.md#原生-memcache-跨重启)
2. 保留原池，使用兼容配置重启推理进程。确认新 worker PID 后重发同一份请求文件，检查响应的 `usage.prompt_tokens_details.cached_tokens`、外部成功读取计数增量和后续输出，再用不同 salt 验证隔离。原生 GLM 同时检查 KV、KPool 和 KDA 边界状态。
3. 运行记录带有池指标地址时，GLM 基准才会写入池指标快照。原生服务可直接在回放前后采样外部池的 metrics 端点。区分 API 就绪、缓存恢复和首条请求完成的耗时。 [原生整机复验](rsi/2026-11-26-glm53-native.md#优化配置的-memcache-跨重启复验) · [vLLM 池复用实测](rsi/2026-11-26-ascend-a3.md#glm-53-flash-与-memcache)
