# kernel 重载时 plan 先于注册单元构建

- 日期：2026-11-10
- 报告人：Codex
- 状态：本分支已修，随 PR #45 验证。
- 引擎：SM90-80GB，PyTorch 2.11.0+cu130；TP1 小 Qwen worker，`cuda_graph`、required、bucket 1/2/4。
- 变更：同时重载 `kernels.reference` 与 `core.plan`，保持 catalog 服务实例。
- 现象：冷启动和 graph 回放正常；重载时 plan 构造失败：`no kernel definition 'gdn.conv1d_fn@1' is registered`；回滚也遇到同样顺序，要求重启进程。OSS command `cmd_20261110T101102Z_graph-focused_ff9522f1`。
- 原因：`_next_ready()` 只按服务就绪和单元名排序。catalog 已存在时，`core.plan` 排在 `kernels.reference` 前；注册不提供独立服务，不能由服务依赖表达“注册已完成”。
- 修复：就绪选择优先非 capture 单元，再选 `captures_graph` 单元；保留原有服务和 optional 约束，不修改事务协议，也不增加空的 kernel readiness 服务。
- 复现与回归：`tests/framework/unit_runtime/test_lifecycle.py::test_capture_waits_for_registrations_on_load_and_reload` 验证真实 Registry 内容；`tests/gpu/test_cuda_graph.py` 检查 kernel 重载后新图可用。真实控制面 apply/撤回由 `test_cuda_graph_apply.py` 覆盖。
- 绕过：首次失败结束测试 worker；修复框架后通过新 OSS release 冷启动，之后的 kernel 修改继续使用 apply。
