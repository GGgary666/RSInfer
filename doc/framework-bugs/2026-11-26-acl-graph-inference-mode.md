# MTP 的 ACL Graph 录制在第二个图切换模式时失败

- 日期：2026-11-26
- 报告人：Codex
- 状态：已修。完整 45 层 TP8 在实机完成录制并通过参考精度 6/6。
- 引擎：完整 GLM-5.3-Flash W8A8，Ascend 910C TP8，grouped GEMM，MTP4，`--acl-graph required`。启动事务中的候选从空引擎录制。
- 现象：Verify 图完成录制和第一次回放。下一个图进入 `capture_begin` 时，8 个 rank 都报 `Inplace update to inference tensor outside InferenceMode`，启动事务回滚。
- 修复：Verify 图和读取其输出的 Draft Extend 图用同一种 `no_grad` 模式录制。它们继续共享图池，Verify 的隐状态仍由 Draft Extend 直接读取。

独立进程的双图复现中，两图同为 `no_grad` 时第二图回放结果逐元素正确。完整模型复测录制了 1、2、4、8、16、32 行的 Verify、Draft Decode 和 Draft Extend。 [首次启动结果](../../../ascend-a3-results/2026-11-26/acl-graph/probe-full8/control/control.json) · [双图复现](../../../ascend-a3-results/2026-11-26/acl-graph/graph-pool-no_grad.log) · [完整模型复测](../../../ascend-a3-results/2026-11-26/acl-graph/probe-full8-recheck/accuracy.json)
