# 进程里一次占住 GIL 超过 `ipc.peer_timeout_s` 的调用，会让对端以为它死了

- 日期：2026-11-10
- 报告人：Claude Opus 5.5 / Devin 在 SM100 上复验 PR #45 时碰到现象；Claude Fable 5.2 在 CPU 上复现并找到原因
- 状态：触发它的那一处已修（账本不再列出全部空闲项）；机制本身没有改。不是热更新时出现的，放在这里是因为它是框架本体（`core.ipc` 的存活判定）的性质，任何单元都可能再碰到。
- 引擎：
  - SM100：`deploy/qwen38.py serve`，很小的模型（`tests/gpu/test_cuda_graph_apply.py` 的 `deployment()`），`--mem-fraction 0.85`，holder 分出 `kv.paged 158961853`；`--cuda-graph off` 和 `required` 都一样。
  - CPU 复现：同一个小模型，`--device cpu --page-size 4 --max-slots 8 --cpu-bytes $((152 << 30))`，holder 分出 `kv.paged 159383485`（CPU 上这 152 GiB 是 calloc 出来的，不写就不占物理内存）。
- 现象：引擎起来以后的第一个请求返回 400，`results of rank 0 come from a new producer (a new worker process?): what the request had there is gone`；调度宿主的日志是 `core.sched: results of rank 0 come from a new producer (a new worker process?): 1 request(s) ended, step(s) [1] given up`。第二个请求起就正常。CPU 上第一个请求用了 5.6 s 才失败。`--mem-fraction 0.3`（约 5600 万页）不出现。
- 期望：进程在忙不等于进程死了；容量再大，第一个请求也应当正常。
- 原因：
  - 调度器收到 worker 的 `Capacity` 以后，账本的 `Pool.set_capacity` 用 `self.free.extend(range(capacity, items))` 把全部空闲项列出来。1.59 亿项在这台机器上是 4.3 s（2100 万项 0.56 s），是一次 C 调用，中间不释放 GIL。
  - 流的心跳是 Python 线程（`core/ipc/stream.py` 的 `_Heartbeat`，每 `ipc.heartbeat_interval_s` = 0.1 s 写一次时间戳），对端在 `ipc.peer_timeout_s` = 2 s 没看到心跳就判定它失联。GIL 被占住的这几秒里，调度宿主在它的流上一次心跳也写不出去。
  - worker 那边于是把调度器当成换了人，重新接上；调度器看到结果流的生产者代数变了（`_results_moved`），把已经发到那条连接上的请求结束掉。
- 修复：`core.state.alloc` 的空闲项改成一个栈，底下从未发出过的 `0 .. fresh - 1` 只记个数，不列出来（`Free`）；发放的顺序与原来的列表相同。改完以后同样的 CPU 引擎第一个请求 0.1 s 返回。
- 没有改的：心跳线程依赖 GIL。任何单元在有流的进程里做一次占住 GIL 超过 2 s 的调用（把很大的 Python 容器一次建出来、不释放 GIL 的扩展调用）都会有同样的后果。可能的方向是把心跳放到不需要 GIL 的线程里；这一轮没有做。
- 复现与回归：`tests/units/scheduler/test_ledger.py::test_a_pool_of_a_billion_items_lists_none_of_them` 和 `test_a_pool_hands_out_the_items_a_listed_stack_would`（进 CI）。端到端的复现要 152 GiB 的虚拟内存，没有放进测试；命令见上面「引擎」。`tests/gpu/test_cuda_graph_apply.py::test_repeated_kernel_changes_leave_the_device_as_it_was` 原来按卡的 85% 分状态池，在 SM100 上就是这 1.59 亿页；现在最多用 24 GiB。
- 绕过：改之前可以用小一点的 `--mem-fraction`，或者调大 `ipc.peer_timeout_s`。
