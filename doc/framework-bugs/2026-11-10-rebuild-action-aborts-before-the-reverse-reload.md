# 重建派生产物的 action 在 abort 里重建，撞上还没撤回的单元，变更回滚不了

- 日期：2026-11-10
- 报告人：Claude Fable 5.2（审 PR #45 时发现）
- 状态：已修（PR #45）。属于「动作的 abort、派生产物的重建」；变更协议本身没有改。
- 引擎：CPU，`deploy/qwen38.py serve --device cpu ...`（`tests/scenarios/test_m2a.py` 的起法），worker 在 `torch_eager` 平台。
- 变更：把 `engine/kernels/reference/unit.py` 里的 `("gdn.recurrent_update@1"` 改个名字（kernel 单元不再登记 decode 要用的一个 definition），`apply --source`。当时的计划：重载 `kernels.reference`，对 `core.plan` 派发 `rebuild_graphs`。
- 现象：`change.execute_failed: ValueError: gdn.backend = 'auto': no kernel definition 'gdn.recurrent_update@1' is registered`，接着 `change.abort_failed: core.plan.rebuild_graphs: ...` 和 `change.abort_unacknowledged`；`status` 的 `recovery_required` 为 true，之后的 apply 都被拒（`a previous change is unresolved; restart the engine`）。
- 期望：失败的变更撤回，引擎照旧服务。master 上这个变更会提交，第一个请求时 worker 出错，把文件改回去再 apply 就恢复，不需要重启。
- 原因：变更撤回时，action 的 `abort()` 在反向重载之前执行。`rebuild_graphs` 的 `abort()` 重新跑一遍重建，这时目录里还是失败的那个 kernel 单元，重建再次失败。协议期望 action 自己留着旧产物、在 abort 里换回去（场景测试的 `scn.rebuild` 就是这样）；decode graph 新旧两套同时留着太占显存，所以这条路不合适。PR #45 原来只在 `cuda_graph` 平台上让 `core.deps` 改为重载 `core.plan`（按平台名判断），`torch_eager` 上仍然派发这个 action。
- 修复：删掉 `rebuild_graphs`。`core.plan` 用 `ctx.derived(..., rebuild="")` 登记；`core.deps` 对没有 rebuild action 的派生产物重载它的 owner，不看平台。撤回时旧的 plan 在恢复后的 kernel 单元之后重新加载。
- 复现与回归：`tests/scenarios/test_m2a.py::test_a_kernel_change_that_cannot_run_rolls_back_and_the_engine_keeps_serving`（CPU，进 CI）；`tests/units/core/test_deps.py::test_an_artifact_without_a_rebuild_action_reloads_its_owner`。
- 绕过：修之前只能重启引擎。
