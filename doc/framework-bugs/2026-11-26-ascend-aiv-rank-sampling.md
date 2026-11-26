# AIV 默认规约下 GLM 各 rank 的采样结果不一致

单机 TP8、EP8 grouped GEMM、MTP 关闭，以 `HCCL_OP_EXPANSION_MODE=AIV` 新启动完整原生引擎。参考精度的前三条通过。第四条请求在第 137 步触发 `ranks sampled different tokens`。调度器结束该请求并拒绝后续请求，避免各 rank 持有不同历史的状态。该轮没有进入 AISBench。 [启动和精度日志](../../../ascend-a3-results/2026-11-26/glm-native-aiv-tp8.log)

相同实现的 AI CPU 展开已通过完整请求。Huawei 描述了 Atlas A3 默认 AIV AllReduce 的非确定性以及 `HCCL_DETERMINISTIC=true` 的用法。单独用 `AIV` 和 `HCCL_DETERMINISTIC=true` 复测。AIV 是否仍实际派发，需要用 trace 验证。 [CANN 确定性规约说明](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/maintenref/envvar/envref_07_0099.html)

8 个真实 die 上把各 rank 的规约输出逐元素与 rank 0 的广播结果比较。AIV 默认条件下，1 行的 BF16 张量累计有 11,050 个元素不同，3 行有 44,144 个。FP32 在这两组也出现差异。显式启用确定性后，1–4,096 行的 14 组 BF16 和 FP32 均逐元素相同。原生启动配方选用 AIV 时默认启用确定性规约。 [默认展开](../../../ascend-a3-results/2026-11-26/validation/hccl-replica-false.log) · [启用确定性](../../../ascend-a3-results/2026-11-26/validation/hccl-replica-true.log)
