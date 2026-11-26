---
name: rsinfer-ascend-optimize
description: 在 Ascend/CANN 上估算 RSInfer 推理性能上限，分析 trace、TTFT/TPOT 与吞吐瓶颈，用同负载基线验收服务优化。
---

# RSInfer Ascend 推理性能优化

## 固定对照

先确定主指标与延迟、显存、正确性约束。沿用可复现的基线 revision 和实际引擎 identity，固定 checkpoint、数值精度、测试输入、采样、缓存冷热、并发和设备分配。把待验证的改动列成一个变量，其他执行参数作为对照常量。仅需估算时，直接按模型、芯片和工作负载计算。 [实验契约与台账](references/experiment-record.md) · [Ascend 性能上限估算](references/ascend-sol.md)

- Qwen3 用 `deploy/bench_ascend.py qwen`。固定 tokenizer、`--salt` 和 repeat 编号，按请求顺序对比两次 `run-*.json` 的 `output_sha256`。它是流式输出文本的摘要。若验收按 token IDs 或 logits，另用固定请求取得对应结果。 [流式基准实现](../../../deploy/bench_ascend.py)
- GLM 用 `deploy/bench_ascend.py glm53` 指向原生或 vLLM 的运行目录。固定 AISBench 套件与冷暖分组。某些套件将输出目录末级名写进提示词，两次实验用不同父目录、相同末级名，核对生成的数据、暖机正文和 salt。精度验收传 `--accuracy-check` 并检查 `experiment.json` 的 `accuracy_exit_code`，`0` 表示检查通过，`null` 表示未运行。 [AISBench 对照命令与输入复现](../../../doc/2026-11-26-ascend-npu-workflow.md#glm-aisbench-对照)
- 原生热更后用已提交 identity 核对基准产物里的配置。跨引擎比较同时标注 TP/EP/DP、ACL Graph、MTP、KV 精度、token budget 和缓存差异。做单因素归因时，用同一引擎的基线与候选。

## 沿关键路径找瓶颈

有现成 trace 时，先核对 rank、stream、请求阶段和采样窗口，读取 `kernel_details.csv` 与 `trace_view.json`。缺少证据时再用 `torch_npu.profiler` 的 Level1 + PipeUtilization 对真实 worker 短采样，分别覆盖 prefill、稳定 decode 和混合步。用临时 `--source` 加探针。单算子按本机 CANN 支持范围采集，多 rank HCCL 或通信计算重叠看应用级时间线。 [采集命令与配置](../../../doc/2026-11-26-ascend-npu-workflow.md#2-用实际-worker-采集性能)

先对齐各 rank 的时间线，再按实际层和模块聚合，找出慢 rank、异常层和关键路径上的空泡。区分 host launch 间隙、Cube/Vector、HBM/KV 搬运与 HCCL 等待。重叠区段按实际墙钟时间分析。 [CANNBot 性能拆解](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/model/model-infer-perf-breakdown/SKILL.md)

| 证据 | 优先验证的改动 |
| --- | --- |
| host 提交或设备空泡 | 实际 ACL Graph 桶与回放、每步分配或同步、调度与组批。 |
| 计算或搬运集中在少数算子 | 复用 CANN 融合，减少格式转换，按目标形状选择算子。若待优化的是权重或 KV 流量，按约定精度验证布局、量化和转换成本。 [算子接入](../rsinfer-ascend-kernel/SKILL.md) |
| 冷 prefill 或暖缓存差异显著 | 实际新处理的 token、前缀命中、状态恢复与跨域复制的等待。 |
| HCCL 或 rank 偏斜 | 通信尺寸、拓扑、MoE 专家负载，及通信是否挡住下一步计算。 |
| decode 验证或草稿成本高 | 每轮提议和实际接受的 token，草稿、校验和提交各自的关键路径开销。 |

## 验收并记录

一轮验证一个有量化预期的假设。可以热更缩短迭代，正式计时前撤掉探针和临时同步。用冻结的测试独立复测正确性和性能，条件允许时交错跑基线与候选。保留原始样本、中位数、波动和尾延迟。收益贴近波动时追加对照或标记待定。达到用户所定正确性和性能目标后，用干净启动复测拟采纳的服务改动。

记录候选适用的形状、阶段或拓扑，同时保留退化和回退证据。结果超过 SOL 估算时，核对 token 口径、缓存命中、同步、并行重叠和模型假设。达到目标、剩余收益低于可辨识噪声或本轮预算用完时，给出结论和复现路径。 [GLM 实验台账示例](../../../doc/rsi/2026-11-26-glm53-native-rsi-iter.md)
