---
name: rsinfer-ascend-debug
description: 定位 Ascend 上 RSInfer 的输出偏差、缓存恢复错误、ACL Graph 故障、AICore/ACLNN 报错、HBM OOM 或 HCCL 卡住。
---

# RSInfer Ascend 推理诊断

从同一个可复现请求和实际引擎入手。记录 identity、热更事务、日志增量、出错阶段、首个报错 rank/die 及实际执行模式。用授权实例中的临时 `--source` 加探针，排空后加载，复现后撤掉。日志、dump 和临时同步单独采集。 [引擎内检查](../rsinfer/SKILL.md#在引擎里做检查排查和验证)

## 输出不一致时

1. 先用基线自对照校验稳定性，再固定 checkpoint、token IDs、采样、缓存和拓扑，比较首个不同 token 对应的 logits。按实际精度和任务约定误差，并检查 NaN/Inf。
2. 从首次分歧的阶段、层、算子逐级缩小。比较整段与分段 prefill、接续 decode、混合 batch 和分页边界。检查 `block_table`、slot mapping、长度、mask、填充槽、ND/NZ 和 scale。
3. 有状态路径在克隆的 KV/KPool/KDA 状态上运行参考，比较 conv、递归状态和缓存写入，再继续生成。MoE 核对专家映射、top-k、归并次序和各 rank 的路由。 [CANNBot 精度定位](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/model/model-infer-precision-debug/SKILL.md)

## 运行失败或卡住时

| 症状 | 优先采集或对照 |
| --- | --- |
| ACLNN / AICore 错误 | 找最早错误的 rank 和步，记录算子名、CANN 错误码、shape/stride、布局、dtype、设备日志。用短诊断复现把同步点移到可疑调用附近。 |
| OOM 或反复 apply 后 HBM 上涨 | 对照同一物理 die 的设备占用与 worker 内 `torch.npu.memory_allocated()` / `memory_reserved()`，分别量启动、apply 前后和大 prefill。拆权重、KV/KDA、录图池、算子和 HCCL 工作区。 |
| HCCL hang / timeout | 先查其他 rank 是否先退出或报错，再核对组成员、建组/collective 顺序和算子类型、count/dtype。记录跨域路由与端口或链路报错。存活 rank 卡住时保留现场日志。 |
| 仅 Graph 或热更后失败 | 以同输入对照 eager、实际 ACL Graph 回放和干净启动。确认录图 bucket、权重派生缓冲、状态复用、回退和新代码是否生效。 |

原生 GLM 可用 `--acl-graph off` 做 eager 对照，再用原配置重放相同 decode。核对生效的 `plan.cuda_graph`、bucket 和实际回放。Qwen3 当前只支持 eager。回放时 Python 探针放在 executor 的图外侧。需看图内值时从 eager 对照取样，避免在 capture 前向中做 `.item()`、D2H dump 或临时同步。 [ACL Graph 录制问题](../../../doc/framework-bugs/2026-11-26-acl-graph-inference-mode.md) · [HCCL 组重建问题](../../../doc/framework-bugs/2026-11-26-hccl-subgroup-rejoin.md)

修复后用同一请求和已知邻近 case 复验。更新机制的异常记录在 `doc/framework-bugs/`，稳定的回归留在对应的测试层。 [报告格式](../../../doc/framework-bugs/README.md) · [CANNBot 运行时诊断](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/model/model-infer-runtime-debug/SKILL.md)
