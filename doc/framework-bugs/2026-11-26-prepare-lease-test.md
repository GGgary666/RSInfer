# 并行测试中准备阶段租约窗口不足

Ascend 分支的 CPU 全量测试在 `test_a_participant_still_preparing_withdraws_when_the_coordinator_vanishes` 复现一次失败。三个参与者仍完成 `abort` 并恢复原 identity，其中 `host-b/worker/1` 刚好跑完 200 片，走已准备参与者的决议查询分支，触发测试对准备阶段退出的断言。单独复跑通过。

参与者会把准备 slice 本身的耗时加回租约。toy 循环每步睡 5 毫秒，200 片之间的服务时间接近 1 秒租约。协调者停止应答前的初始消息可续期，负载与调度时序决定该参与者能否在租约过期前完成最后一片。

把该场景的准备片数提高到 400，保留原 1 秒租约和 50 毫秒 slice。参与者仍需在准备阶段退出，断言回滚、identity、恢复服务以及 `while preparing` 诊断。修改后并行复测同文件 9/9 通过。
