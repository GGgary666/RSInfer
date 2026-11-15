# 多卡 EP 热改中单个 worker 触发设备致命错误后，其余 worker 在撤回阶段卡死于跨卡集合通信

- 日期：2026-11-16
- 报告人：Antigravity（PR #66 多卡热改开发与验证）
- 状态：未修
- 引擎：在 SM100 节点 `n1` 的 4 张卡（GPU 4, 5, 6, 7）上以 `ep-live`（身份 `d92b2330547f4921`）启动 4-rank EP 引擎：
  `python deploy/agxinfer.py serve /models/DeepSeek-V4.1-Flash-1017 --source ep-live --run run-ep --gpus 4,5,6,7 --port 28166 --backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 64 --disable-radix-cache --detach`（`platform = cuda_graph`，`comm.size = 4`，`comm.device_backend = nccl`，`comm.timeout_s = 1800.0`，`change.execute_timeout_s = 3600.0`）
- 变更：`python deploy/agxinfer.py control apply --run run-ep --source ep-work --intent "attention fusion + EngramBuffer + prefill warmup + symm hot-apply fix"`（变更 ID 10）；修改了 `kernels.ds1017`、`kernels.tilelang`、`kernels.deepinfer` 与 `layers.ds1017`，计划在 `worker/0..3` 上原地重载 `core.plan`、`kernels.deepinfer`、`kernels.ds1017`、`kernels.tilelang`、`layers.ds1017`
- 现象：
  1. 在新代码加载阶段，`layers.ds1017` 的单卡预热草稿逻辑传给 `deep_select.topk` 的 `cand` 含重复块号，导致 `worker/2` 读到旧缓冲区未初始化的 `NaN` 字节并触发 PTX `trap;`（日志打印 `[topk_select] NaN detected. Calling trap; which will result in "unspecified launch failure"`），CUDA 上下文损坏。
  2. `worker/2` 的 `UnitRuntime._reload` 失败并进入 `UnitRuntime._rollback`，回退中的 `core.plan` 因同一粘性 CUDA 错误立即再次失败，`worker/2` 向 supervisor 返回 `Failed(code="escalate.restart_process")` 并回到 `SyncRpcServer.poll` 空闲等待。
  3. 其余三个 rank（`worker/0`、`worker/1`、`worker/3`）未读到 `NaN` 字节，进入 `core.plan`（或随后的 `_rollback` -> `core.plan` -> `eager_padding_step` -> `Layers.fetch` -> `Group.gather` -> `dist.all_gather_into_tensor`），`py-spy` 栈显示三者全部阻塞在 `torch/distributed/distributed_c10d.py:4394 (all_gather_single)` 等待已经退出 `on_execute` 的 `worker/2`。
  4. 控制面虽然已经收到 `worker/2` 的 `Failed`（且带 `escalate.restart_process`），但 `execute` 阶段仍在等待全部参与者返回（最长受 `comm.timeout_s = 1800s` / `change.execute_timeout_s = 3600s` 限制），导致 `control apply` 长时间挂起，引擎整体处于 `busy: true` 不可服务状态。
- 期望：
  多 rank 协同 worker 组在 `execute`（重载或回退重录图）期间若包含跨 rank 集合通信（如 `eager_padding_step` 中的 EP `all_gather`、`SymmBuffer` / `EngramBuffer` 构造），一旦任意一个参与 rank 已经向协调者返回 `Failed`（尤其是 `escalate.restart_process`），协调者不应继续静默等待其余 rank 卡满 30–60 分钟的集合通信超时，而应尽快中止本轮变更并对失步的通信组执行进程级重启恢复（与冷启动或进程崩溃恢复一致）。
- 复现：
  在 `comm.size >= 2` 且 `core.plan` 重载会执行跨 rank `eager_padding_step` 的拓扑上，让仅 `rank == 1` 的单元在 `resolve()`（进入集合通信前）抛出异常并使其 `_rollback()` 也抛出异常；`rank == 0` 会在 `eager_padding_step` 的集合通信处一直阻塞到 `comm.timeout_s` 超时。可作为多进程集成测试放入 `tests/framework/cluster/`。
- 绕过：
  修复了单元侧 `layers.ds1017` 预热入参（候选块填 `[0, -1, ...]`，消除 `trap;` 根因），并用 `agxinfer stop --timeout 10` 重启一次 `run-ep` 清理损坏的 CUDA 上下文与失步的进程组；此后继续在同一 4 卡引擎上完成了后续全部热改验证，未再复现。
