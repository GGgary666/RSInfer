# Ascend 910C 推理调优与验证

适用于在 RSInfer 的 `dev/ascend_a3_gg` 分支开发 NPU 模型、算子和缓存路径。 [启动参数](../deploy/README.md#ascend-910c) · [热更事务](../.claude/skills/rsinfer/SKILL.md#在跑着的引擎上热改)

| 路径 | 确认实际运行的实现 |
| --- | --- |
| `rsinfer.py serve --device npu` | RSInfer 的 Ascend worker。Qwen3 走 `kernels.ascend`，GLM-5.3-Flash 走 `models.glm53` 与 `layers.glm53`，默认要求 decode ACL Graph。以运行目录的配方、各 worker 的 `core.plan` 记录和实际图回放数量核对派发。 |
| `rsinfer.py glm53 serve` | vLLM Ascend GLM 对照部署。用 `glm53 status`、`launch.json`、`ready.json` 和服务日志核对 TP/DP/EP、MTP、ACL Graph、MemCache 配置。 |

## 1. 核对设备和运行环境

从启动 worker 的同一环境检查驱动、CANN、PyTorch 和 `torch_npu`。环境尚未加载时，先 source 该 CANN 安装的 `set_env.sh`。本机的已测版本组合见 [Ascend 910C 部署与 RSI 实测](rsi/2026-11-26-ascend-a3.md)。

```sh
npu-smi info
npu-smi info -m
npu-smi info -l  # 多 die 时同时核对拓扑
command -v msprof
python - <<'PY'
import os
import sys
from importlib.metadata import version

print("python", sys.executable)
for package in ("torch", "torch-npu"):
    print(package, version(package))
for key in ("ASCEND_HOME_PATH", "ASCEND_OPP_PATH", "ASCEND_CUSTOM_OPP_PATH", "ASCEND_RT_VISIBLE_DEVICES"):
    print(key, os.environ.get(key))
PY
```

- 根据映射和进程占用选择可用物理 die，再传给 `--npu` 或 `--npus`。节点代理设置 worker 的设备可见性后，单 die worker 在本进程内使用 `npu:0`。多 rank 排查同时记录物理 die、worker rank、调度域和通信组。
- 出现自定义算子找不到或加载失败时，查看实际 worker 启动时的 `ASCEND_CUSTOM_OPP_PATH`、扩展出处和 OPP 版本。原生 GLM 配方在设备初始化前准备自定义算子路径。
- A3 的 `npu-smi` 芯片名可能显示 `Ascend910`。选算子或编译目标时，核对 `asys info -r=hardware` 的芯片信息和实际 CANN 支持的 SoC。CANNBot 的设备预检还提供基于 asys 和 DSMI 的架构探测。 [ascendc-env-check](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/ops/ascendc-env-check/SKILL.md)

## 2. 用实际 worker 采集性能

先查运行入口是否已有 profiler。RSInfer 原生路径需要临时探针时，把它写进独立的 `--source` 工作区，通过 `control apply` 加载。探针放在需要观察的层或 executor 的步循环上，采集器随单元实例持有。每份 trace 标记引擎 identity、phase、rank 和 die。用短窗口分别采样 prefill 和稳定 decode，混合步单独标记，将首次 JIT、图录制和服务预热放在稳定窗口之外。

采集器配置示例。调用点提供 `trace_dir`、`rank` 和 `active_steps`。真实执行一步后调用 `prof.step()`，退出前让 schedule 完成 active 窗口和保存回调。

```python
from torch_npu import profiler as p

prof = p.profile(
    activities=[p.ProfilerActivity.CPU, p.ProfilerActivity.NPU],
    schedule=p.schedule(wait=0, warmup=0, active=active_steps, repeat=1),
    on_trace_ready=p.tensorboard_trace_handler(str(trace_dir), worker_name=f"rank-{rank}"),
    record_shapes=True,
    experimental_config=p._ExperimentalConfig(
        profiler_level=p.ProfilerLevel.Level1,
        aic_metrics=p.AiCMetrics.PipeUtilization,
    ),
)
```

- 算子或 HCCL 的独立复现可用 `msprof --runtime-api=on --task-time=on --hccl=on --aic-metrics=PipeUtilization --output="$TRACE_DIR" python "$REPRO_SCRIPT"`。按本机 `msprof --help` 核对选项。独立计时使用同口径的预热和重复次数，报告真实调用点的重排、拷贝和同步开销。多 rank 通信和计算重叠用完整应用采集。 [ops-profiling](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/ops/ops-profiling/SKILL.md)
- 采集后确认 `ASCEND_PROFILER_OUTPUT` 的 `kernel_details.csv`、`trace_view.json` 及所需的流、形状和流水利用率字段。解析失败时先核对采集窗口、解析日志和输出目录所在文件系统。保留原始采集，必要时在本地盘重新解析。 [model-infer-profiling](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/model/model-infer-profiling/SKILL.md)
- 沿关键路径分辨 CPU 调度、launch 间隙、计算、KV 或权重搬运、HCCL 与等待。按 rank 和 stream 看时间线，再按 KDA、稀疏 MLA、MoE 等模块聚合。并发区间按 wall time 计，单列可重叠的通信与计算，查看慢 rank 和离群层。探针和 profiler 退出后，再做正式的 API 性能测试。 [model-infer-perf-breakdown](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/model/model-infer-perf-breakdown/SKILL.md)

## 3. 接入候选与验证正确性

在真实调用点记录 shape、stride、dtype、ND/NZ 或 paged layout、序列长度元数据、量化 scale/offset 和原地写入。优先查当前镜像里的 `torch_npu` 或已加载扩展的 API 与算子实现，核对 910C 上支持的布局和范围。需要融合或换算子时，把必要的格式转换和额外内存算进收益。 [model-infer-fusion](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/model/model-infer-fusion/SKILL.md)

1. 固定 checkpoint、token IDs、采样参数和参考实现。先做基线自对照，再比候选的张量、logits 和输出。检查 NaN/Inf，按 dtype 和算子预先约定误差判定。
2. 从出现偏差的阶段向下收窄，比较单段与分段 prefill、接续 decode、混合批次和分页边界。分页 attention 核对 `block_table`、slot mapping、有效长度、mask 和填充槽。涉及 KDA 或混合前缀时，在克隆的状态上比 conv、递归状态与 KV，验证恢复后继续生成。 [model-infer-precision-debug](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/model/model-infer-precision-debug/SKILL.md) · [model-infer-kvcache](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/model/model-infer-kvcache/SKILL.md)
3. 路由、融合和分布式候选覆盖多个实际 batch、序列长度与 rank 分布。对 MoE 检查专家映射、top-k 与加权归并。性能用例来自请求中的常见形状，并覆盖邻近形状和边界。
4. 对同一个引擎运行 `control check`、`control apply --dry-run`，然后用 `apply --wait-empty` 提交。记录前后 identity、worker PID、受影响的单元和暂停时间。按规划决定热更或重启，变更了进程或状态时测量对应的权重装载、缓存恢复和首请求。候选回退后用同一请求验证服务。

| 现象 | 排查入口 |
| --- | --- |
| ACLNN 报错或 AICore timeout | 从最早的出错 rank 和步开始看，记录算子名、错误码、实参和对应的 CANN 日志。在短的诊断复现中把同步点放到可疑调用附近，找出最先出错的调用。 |
| OOM 或热更后内存上涨 | 对照 `npu-smi` 的 HBM 占用和 `torch.npu.memory_allocated()` / `memory_reserved()`，按 rank 区分权重、KV/KDA、临时算子和通信工作区。记录启动、apply 前后和首个大 prefill 的峰值。 |
| HCCL 卡住或超时 | 先看其他 rank 是否已报错或退出，再核对通信组及各 rank 进入 collective 的顺序、count 和 dtype。对 MoE 额外核对涉及的 rank 与路由分布。 |

同步、逐层 dump 和 profiling 使用独立诊断轮次。 [model-infer-runtime-debug](https://raw.gitcode.com/cann/cannbot-skills/raw/5cdfa1854b7e656222cf67d6dbd76533cd683d30/model/model-infer-runtime-debug/SKILL.md)

## 4. 端到端对照与采纳记录

Qwen3 可用 `deploy/bench_ascend.py qwen` 做小模型验证。GLM 使用 `deploy/bench_ascend.py glm53` 指向各自的运行目录，用同一套 AISBench 数据生成流程和精度检查跑 16K 与 60K。固定样本选择、模板、并发、生成长度、停止条件和冷/暖缓存流程。保留每轮数据、salt 和预热结果，沿用参考套件的独立冷前缀与共享前缀预热。需要逐 token 判定时，另外重放完全相同的 token IDs。记录 DP/TP/EP、Eager/Graph、MTP、token budget 和缓存配置，拓扑不同的结果注明差异。

```sh
python deploy/bench_ascend.py glm53 --run "$ENGINE_RUN" --suite "$AISBENCH_SUITE" \
  --accuracy-check "$ACCURACY_CHECK" --out "$BENCH_OUT"
```

- 汇报成功请求数、输出 tok/s、TTFT 与 TPOT 的均值和 P90、实际输入与生成 token 数、精度结果和缓存命中。明确讲清基线、候选和实际启用的执行路径。单算子或裁剪模型结果用于定位；用完整服务的对照数据决定采纳。
- MemCache 跨重启单独验证。先在缓存未命中时提交固定前缀并确认写入，保持 MetaService 和 Pool 运行，重启推理进程，再以相同 token IDs 和 salt 提交。核对新 worker PID、原池 PID、该次请求的外部读取与命中增量、恢复长度和输出正确性。更换 salt 可验证隔离。原生混合模型的恢复证据需覆盖 KV 和匹配的 KDA 边界状态。分别记录池复用、服务就绪和首条请求完成的耗时。
- 在 `doc/rsi/` 保留每次实验的假设、profile 和评测路径、变更 identity、精度及性能实测、采纳或拒绝理由。保留性能退化、精度失败或未能复现的记录。已有对照见 [Ascend 910C 实测](rsi/2026-11-26-ascend-a3.md) 和 [原生 GLM 优化记录](rsi/2026-11-26-glm53-native.md)。
