# 启动提交时 worker 因意外的 RPC 帧数量退出

- 日期：2026-11-26
- 报告人：Codex
- 状态：已修。同配置 16-die 重试以 identity `75a77b5906c64c8b` 提交，所有 worker 恢复并通过六项精度检查。
- 引擎：原生 GLM，Ascend 910C，DP2 × TP8，DFC，MTP4，ACL Graph required。16 个 worker 已完成图录制。
- 现象：rank 5 在处理 RPC 时直接解包 multipart，报 `ValueError: too many values to unpack` 并退出。控制面已作出启动提交决定，三次恢复确认均未得到回复，进入 recovery required。
- 期望：接收端仅将符合两帧协议的请求交给解码器，意外帧的数量不会影响 worker 存活。两种接收循环均增加帧数检查。
- 回归：`test_sync_server_discards_an_unexpected_frame_envelope` 用替身 socket 验证意外封套被丢弃；正常往返用同文件的 RPC 测试覆盖。

[首次失败事务和 worker 日志](../../../ascend-a3-results/2026-11-26/acl-graph/dp2-dfc-first) · [实机复测](../../../ascend-a3-results/2026-11-26/acl-graph/dp2-dfc.log) · [回归日志](../../../ascend-a3-results/2026-11-26/acl-graph/cpu-first-regressions.log)
