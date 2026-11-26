# Ascend 910C / CANN 文档导航

第一次运行 RSInfer，可以从单 die 的 Qwen3 开始，先确认 worker 就绪并拿到一条补全结果。已有运行实例时，按任务查下表。 [Qwen3 部署与首条请求](../deploy/ascend.md#rsinfer-原生-qwen3)

| 任务 | 命令与实操 | 对应 skill |
| --- | --- | --- |
| 部署、核对物理 die 和调度域 | [安装依赖](../deploy/ascend.md#安装依赖) · [设备预检](2026-11-26-ascend-npu-workflow.md#1-核对设备和运行环境) · [原生 GLM](../deploy/ascend.md#rsinfer-原生-glm-53-flash) · [vLLM Ascend 对照](../deploy/ascend.md#vllm-ascend-glm-53-flash-对照) | [环境与运行](../.claude/skills/rsinfer-ascend-runtime/SKILL.md) |
| 估算上限，定位 TTFT、TPOT 或吞吐瓶颈 | [SOL 估算](../.claude/skills/rsinfer-ascend-optimize/references/ascend-sol.md) · [worker 性能采集](2026-11-26-ascend-npu-workflow.md#2-用实际-worker-采集性能) | [推理性能优化](../.claude/skills/rsinfer-ascend-optimize/SKILL.md) |
| 接入 CANN 算子、验证融合或 tiling | [实际派发与正确性回归](2026-11-26-ascend-npu-workflow.md#3-接入候选与验证正确性) · [GLM 融合与回退案例](rsi/2026-11-26-glm53-native.md#融合-kda-输出-rmsnorm-对照) | [算子接入](../.claude/skills/rsinfer-ascend-kernel/SKILL.md) |
| 排查错误 token、ACL Graph、HCCL 或热更故障 | [按症状查报告](framework-bugs/README.md#ascend-与-cann-案例) · [引擎内探针与恢复](../.claude/skills/rsinfer/SKILL.md#在引擎里做检查排查和验证) | [推理诊断](../.claude/skills/rsinfer-ascend-debug/SKILL.md) |
| 验收服务性能、精度或跨重启缓存命中 | [Qwen3 流式基准](2026-11-26-ascend-npu-workflow.md#qwen3-流式基准) · [GLM AISBench](2026-11-26-ascend-npu-workflow.md#glm-aisbench-对照) · [MemCache 复验](2026-11-26-ascend-npu-workflow.md#memcache-跨重启验收) | [实验记录模板](../.claude/skills/rsinfer-ascend-optimize/references/experiment-record.md) |

## 查找已有实测

- **Qwen3 与 vLLM Ascend 对照**。单 die Qwen3 的 BF16 输出与性能、双副本验证，以及 vLLM GLM 的冷暖组与池复用结果。 [910C 部署与实测](rsi/2026-11-26-ascend-a3.md)
- **原生 GLM 基线与机制验证**。按实验查 ACL Graph 回放、跨域前缀复制、CANN profile 和完整模型 MemCache 恢复。 [原生执行优化记录](rsi/2026-11-26-glm53-native.md)
- **原生 GLM 后续迭代**。16-die 双域配置下每轮的 identity、收益、精度、保留或回退决定及复现参数。 [逐轮实验台账](rsi/2026-11-26-glm53-native-rsi-iter.md)

以上记录对应 2026-11-26 的实验环境。配置、结论和汇总表随仓库保存。打开原始 trace、CSV 和日志链接时，需在仓库同级放置配套的 `ascend-a3-results/2026-11-26/` 产物目录。
