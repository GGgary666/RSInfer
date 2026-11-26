# 热更后的 AISBench 启动元数据过期

同一运行目录通过 `control apply` 将原生 GLM 的 `spec.num_steps` 从 3 改为 4。执行事务与 worker 正确进入 MTP4。AISBench 输出的 `experiment.json` 仍记录 MTP3，因为它读取的是 `serve` 写入的启动配置。吞吐和延迟结果来自实际运行的候选。受影响的是候选配置的归档。 [评测元数据](../../../ascend-a3-results/2026-11-26/glm-native-aisbench-tp8-combined/experiment.json) · [运行中的候选事务](../../../ascend-a3-results/2026-11-26/glm-native-mtp3-cache-hot-recheck/glm-native-aisbench-tp8-combined/candidate-apply.json)

原生评测在开跑前读取最近一次已提交的 identity，从其 manifest 记录配置和拓扑。用实际控制日志的副本分别读回候选与卸载后的 identity。候选正确记录 4 个 MTP 步、5-token verify 窗口，卸载后读回 3 个 MTP 步。原有结果文件保留，候选事务是该轮配置的记录。 [复测日志](../../../ascend-a3-results/2026-11-26/validation/aisbench-live-identity.log)
