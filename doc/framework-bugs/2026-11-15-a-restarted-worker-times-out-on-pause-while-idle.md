# 重启过 worker 之后，只要还没有处理过新请求，下一次 in-place `apply` 就会卡在 `Pause` 30 秒超时撤回

- 日期：2026-11-15
- 报告人：Devin
- 状态：未修
- 引擎：`deploy/rsinfer.py serve` 起单机 Qwen3.8（frontend + sched + worker，CPU `torch_eager`），先完成至少一条请求（令调度宿主 `host.epoch = E >= 1`）。
- 变更：
  1. 提交一次带 worker 重启的变更（例如 `control apply --topology <file>` 修改 worker 角色 `env`）——提交成功，`host-a/worker/0` 重启为新进程；
  2. **中间不发任何推理请求**，直接提交一次涉及 worker 的 in-place 变更（例如向工作区新增一个 `process=["worker"]` 的单元并 `control apply --source <workspace>`，或修改已有 worker 单元代码）。
- 现象：第 2 次 `apply` 在 `pause` 阶段等待 30 秒后 `aborted`，诊断为：
  - `ipc.timeout`（subject 为重启后的 worker 地址）：`no reply to Pause within 30.0s`（`the peer may be lost`）。
  - 在 CI 运行 `43701136497`（`tests/scenarios/test_m2a.py::test_a_unit_added_to_the_workspace_loads_at_the_next_apply_and_one_removed_unloads`）中实测触发。
- 期望：照 `doc/specs/change.md` §4.2 第 4 条（"worker 报告的是它实际做完的那一步的编号，重启后直接接入当前编号，不从零重新数"），空闲状态下重启过 worker 后，紧接着的 in-place 变更应能立即完成 `Pause` 并提交，不需要先人工发一条请求"喂"出新的 step epoch。
- 根因定位：
  - `Coordinator._pause`（`engine/core/change/coordinator.py:491-494`）先暂停 `leads_epochs` 的调度宿主，拿到 `Paused.epoch = E`（即重启前最后退休的 step 编号 `self.table.last_retired = E >= 1`），随后向跟随者发送 `Pause(run.id, after=E)`。
  - 重启后的新 worker 进程 `host.epoch` 初始为 `None`，`WorkerLoop.epoch` 也为 `None`（只在 `WorkerLoop._flush` 写出 `epoch > 0` 的 step 结果时才更新，`Capacity` 记录的 `epoch == 0` 不更新 `self.epoch`）。
  - 当引擎处于空闲、尚未跑过新 step 时，调度宿主已经暂停不再发新 step，新 worker 永远不会收到 `epoch >= E` 的 `STEP_PLAN`；`Participant._reached(after=E)`（`engine/core/change/participant.py:554-560`：`self.host.epoch is not None and self.host.epoch >= after`）恒为 `False`，`pause_reply` 一直挂起直到 `rpc_timeout_s`（30s）超时撤回。
- 复现：
  1. `serve` 启动引擎并发一条 `/v1/completions` 请求（使 `sched` 的 `epoch >= 1`）；
  2. `control apply --topology` 改 worker 的 `env` 触发 worker 重启（`committed`）；
  3. 不发请求，直接 `control apply --config '{"plan.cuda_graph": "off"}'` 或 `--source` 触发 worker 参与的变更，必现 30s `Pause` 超时。
- 绕过：在 worker 重启后跑过至少一条请求（使新 worker 处理一个 `epoch > E` 的 step，将 `host.epoch` 推进到 `>= E`），再做下一次 in-place `apply`。彻底修复可在框架或循环层同步空闲 epoch（例如重启进程 `Resume` 时或 `Capacity` 握手中接入当前 epoch，需单独评估是否触碰框架本体与 formal 模型）。
