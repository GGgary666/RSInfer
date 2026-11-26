---
name: rsinfer-ascend-runtime
description: 在 Ascend 910C 配置 RSInfer worker 的 CANN/OPP 环境，启动原生 Qwen3、GLM 或 vLLM Ascend 对照，核对拓扑、执行模式与 MemCache 跨重启恢复。
---

# RSInfer Ascend 环境与运行

## 确认 worker 环境

- 在启动 worker 的同一个 Python / 容器中检查驱动、CANN Toolkit 和 OPP、自定义算子包、`torch` / `torch_npu` 及其导入路径。需要加载 CANN 时，使用当前安装的 `set_env.sh`。GLM 原生配方依赖带自定义算子的 `vllm_ascend` 和 `fla_npu`，检查 worker 的 `ASCEND_CUSTOM_OPP_PATH` 和实际加载的扩展。 [预检命令](../../../doc/2026-11-26-ascend-npu-workflow.md#1-核对设备和运行环境)
- 用 `npu-smi info` 看设备健康和占用，`npu-smi info -m` 看物理 die 映射，`npu-smi info -l` 看拓扑。A3 的显示名可能只是 `Ascend910`，涉及编译目标或 API 支持时，用 `asys` 或 DSMI 核对具体 SoC。RSInfer 的 `--npu` / `--npus` 选物理 die，worker 内的设备序号受可见性映射控制。 [CANNBot 环境预检](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/ops/ascendc-env-check/SKILL.md)
- 如果启动 shell 可以导入依赖而 worker 报 TBE 或 OPP 缺失，核对节点代理给该实例的 Python 搜索路径和启动环境。worker 从按 identity 物化的快照加载引擎。 [TBE 加载故障记录](../../../doc/framework-bugs/2026-11-26-ascend-runtime-python-path.md)

## 启动与确认

| 目标 | 从哪里启动 | 怎样确认 |
| --- | --- | --- |
| 原生 Qwen3 | `serve` 按 checkpoint 选配方。`--npus` 每个 die 放一份完整模型，独立调度。 | 当前 NPU 配方只支持 eager。用短请求验证输出，在 worker 查看分页算子是否命中。 |
| 原生 GLM-5.3-Flash | 同一 `serve` 选 `glm53-native`。`--glm-dp-size` 须整除 rank 数，每个调度域内部做 TP/EP，每个 die 一个 rank。 | 默认 `--acl-graph required`，prefill 走 eager。对照设备放置和 HCCL 组，用代表请求的 trace 或计数确认 decode 回放。启用 MTP 时分别看 draft 和 verify。 |
| vLLM Ascend GLM 对照 | `deploy/rsinfer.py glm53 serve` | 用 `glm53 status`、`launch.json`、`ready.json` 和服务日志核对最终参数。配置变更通过 `glm53 restart` 生效。 |

原生服务启动前可用 `python deploy/rsinfer.py serve "$MODEL" --device npu --npus "$DIES" --run "$PLAN_DIR" --dry-run`，将本次预览写到独立目录。检查 `topology.json` 的 placement、mesh 和 `config.json` 的 `sched.domains`、`plan.cuda_graph`。服务启动后以实际 identity、worker 日志和请求为准。 [部署命令](../../../deploy/ascend.md)

## 热更与缓存恢复

原生引擎按通用流程检查计划、排空并重放请求。`platforms.ascend` 属于框架，改动后需要重启。 [热更与恢复命令](../rsinfer/SKILL.md#在跑着的引擎上热改)

验 MemCache 跨重启时，先用一个固定前缀确认写入并记录池 PID。原生 GLM 用 `cache-save` 和 `--prefix-manifest`，保持模型、状态格式和 rank 划分兼容。保持原池运行，确认新的推理 PID，再重放相同 token IDs 和 salt。以该次请求的外部读取与命中增量、恢复的 token 长度和正确输出为证据。混合模型还要验证 KV、KPool 与 KDA 边界状态在恢复后能继续生成。 [MemCache 复验](../../../doc/2026-11-26-ascend-npu-workflow.md#memcache-跨重启验收)
