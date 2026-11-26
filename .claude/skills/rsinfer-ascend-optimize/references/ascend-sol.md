# Ascend 推理性能上限估算

用于在改动前选瓶颈，或解释实际结果距同一模型、设备与负载的 Speed-of-Light（SOL）还有多少空间。为理论规格、微基准校准结果和服务实测分别注明来源。

## 选择计量单位

- Prefill 选一次请求或 chunk，记录实际新处理 token 数、有效上下文和缓存恢复量。Decode 选一次调度步或一个稳定窗口，记录同时推进的请求数与实际提交的输出 token。MTP 按接受并提交的 token 算有效吞吐，另记提议、校验开销和接受率。
- 写清是单 die、一个 TP/EP 调度域，还是整机的量。对在线负载，同时跟踪端到端 TTFT、TPOT 与排队。各域只有在负载、并发和共享资源允许时才可累加吞吐。用真实形状和各 rank 的分片估算。

## 建立可审计的下界

| 项 | 分子 | 分母或测量 |
| --- | --- | --- |
| Cube 计算 | 实际 GEMM 的 FLOPs 或整数运算量。普通 GEMM 可用 `2*M*N*K`。 | 每 die、目标 dtype 和代表性形状上的有效算力。W8A8 的乘加与量化/反量化路径分开估。 |
| Vector 计算 | Norm、激活、量化、索引等目标算子的工作量。 | 同类运算和形状的微基准速度，或分阶段测出该段耗时。 |
| HBM | 权重及 scale、KV/索引、状态和关键中间量的读写字节。 | 同 die 的实际拷贝带宽，注明读写口径，避免把缓存内通过量当成 HBM 带宽。 |
| HCCL 与缓存传输 | 每组、每步的有效载荷及次数，包括独立的缓存传输。 | 同拓扑和消息尺寸测出的 collective 或传输时延。TP all-reduce、reduce-scatter、EP all-to-all 和跨域 KV 各自量测。 |
| Host / MTE 等其他成本 | 提交、依赖和不能隐藏的数据搬运。 | 从代表性 trace 确定其暴露部分。片上搬运用相应资源指标定位。 |

对可能并行的一段，用独立资源耗时中的最大值作为乐观下界。沿同一依赖路径累加确实串行的段，并考虑 collective 要等的慢 rank。共享资源上的工作先在同一口径合计。以 `有效 token 数 / 步时下界` 估算吞吐上限，以同口径的 `实测吞吐 / 估算吞吐上限` 表达距离。通信时延包含启动成本时，避免再加一次同样的开销。

仅有型号规格时，可以先给理论值或区间，标注未校准的项。要形成可用目标，再用同 CANN、设备状态、精度与拓扑复测代表性的 GEMM、HBM 搬运和 HCCL。借 trace 判断 overlap 和实际瓶颈。以不带 profiler 的 API 性能测试作为验收数据。

## 用 RSInfer 的实际路径修正

- Qwen3 以 GQA 的 KV 头数、页大小、有效长度和实际分配的页计数。共享前缀减少新 prefill token，已缓存 token 的 decode attention 读流量仍要按实际路径计算。
- GLM 的混合层按实际份额分别估算 KDA、稀疏 MLA/DSA、MoE。按 KV、KPool、索引和 KDA 状态各自的格式估算搬运和容量。MoE 按每个 rank 实际路由的 token、命中的专家与副本计活跃流量，另外计入本 rank 常驻的全部权重。
- 显存可行性按每个 rank 的权重、KV 页与状态槽、录图缓冲、算子和通信工作区预算，取先耗尽资源的 rank。再根据实测的 TPOT、TTFT 与尾延迟选并发和 batch。
- 优先验证能从关键路径移除的耗时。若某段在 trace 上已被遮蔽，单算子加速的端到端收益预期也应下调。

没有可信的流量或操作计数时，给出范围、主要假设和可证伪的下一项测量。占比异常时先查单位、计时同步、缓存命中、MTP 提交口径与慢 rank。 [CANNBot 并行策略分析](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/model/model-infer-parallel-analysis/SKILL.md) · [SRE 第一性原理](https://github.com/HappyFox001/Super-Reasoning-Engine/blob/v0.1.0/references/01-first-principles.md)
