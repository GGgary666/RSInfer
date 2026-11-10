# kernel 热改后重录 graph，旧图的显存没有还给设备，几次以后 apply 失败且回滚不了

- 日期：2026-11-10
- 报告人：Claude Fable 5.2（审 PR #45 时发现）
- 状态：已修（PR #45）。属于「单元为热更新实现的东西出了错」：`core.plan` 的重录。
- 引擎：RTX 4090 D，PyTorch 2.11.0+cu130；`deploy/qwen38.py serve --model <27B 结构的 4 层> --cuda-graph auto --max-rows 8 --max-context 1024`，worker 在 `cuda_graph` 平台。
- 变更：给 `engine/kernels/reference/ops.py` 末尾加一行注释，`apply --source`；重复。计划是重载 `kernels.reference` 和 `core.plan`。
- 现象：
  - 默认 `--mem-fraction 0.85`：第 1 次就失败。新的 plan 录图时 `CUDA out of memory`；反向重载时旧的 plan 也录不出来，`escalate.restart_process: units did not come back: ['core.plan', 'core.worker', 'executors.pytorch']`，`change.recovery_required`，请求超时。
  - `--mem-fraction 0.6`：每次 apply 以后 nvidia-smi 多 1390 MiB，第 6 次 `MemoryError: not enough free memory for plan.graph_memory_bytes`，同样回滚不了。
  - worker 进程内重载这两个单元：`memory_allocated` 不变，`memory_reserved` 每次多 1.36 GiB。
- 期望：重录以后占用回到重录之前；失败的变更能撤回（`doc/specs/cuda-graph.md` §7、skill 里「资源在多次 apply 之后一直涨」）。
- 原因：`CUDAGraph.reset()` 以后，图的私有池里的块留在 PyTorch 的缓存里；下一批图录进自己的新池，用不了这些块，只能再向驱动要。原来的预检还用 `torch.cuda.mem_get_info`，缓存里的块不算空闲。
- 修复：`Plans.close()` 在图都关闭、固定 buffer 都释放之后 `torch.cuda.empty_cache()`；一批图共用一个池；去掉预检，改为量图留下的 `memory_reserved`。
- 复现与回归：`tests/gpu/test_cuda_graph.py::test_overlap_and_repeated_rebuild_keep_outputs_and_memory`（重载 10 次，断言 allocated、reserved 和设备空闲都不变；原来只断言 allocated，泄漏时它是平的）；`tests/gpu/test_cuda_graph_apply.py::test_repeated_kernel_changes_leave_the_device_as_it_was`（真实引擎，默认 `mem_fraction`，10 次 kernel 变更，中间一次失败并撤回）。
- 绕过：没有；修之前只能重启引擎。
