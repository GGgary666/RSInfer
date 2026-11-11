# 出错以后改好就能恢复、让出显存、新的命令行（2026-11-11，RTX 4090 D）

这一组改动来自一次讨论：用热更新开发时哪些地方别扭，热更新的 bug 哪些修了、哪些没修。用户定下的几件事，这里是第一个 PR 的部分，都不改框架本体：

- worker 出错以后改好代码再 apply 也不恢复（[a-repaired-worker-stays-stopped](../framework-bugs/2026-11-11-a-repaired-worker-stays-stopped.md)）：用户同意的两个方向，一起做了；
- 让出显存（sleep mode），让同一张卡能边开着引擎边跑 GPU 测试；
- 新的命令行 `deploy/agxinfer.py`，取代 `deploy/qwen38.py`：起停引擎和改引擎分开，带上 `stop`；
- 临时检查放在另一份工作区里，再加一个扫描测试兜底。

另外两件要改框架本体（准备阶段编译、心跳核实），在第二个 PR 里。

## 结论

1. **出错以后，把代码或配置改好 apply 就恢复服务，不用 `--force`，也不用重启。** 出错那一刻，引擎里的请求立刻全部回错误：在途的请求 0.05 秒就回了，以前会一直挂着。真的 CUDA OOM 也一样能救回来。
2. **`sleep` 把 4 层引擎的显存从 21,482 MiB 降到 962 MiB，2 秒；`wake` 3.4–3.6 秒，醒来输出逐字相同。** 睡着的时候，`tests/gpu/` 的 8 个文件（50 项）在同一张卡上全部通过。
3. **新命令行跑通了真引擎和 CPU 上的端到端测试。** `stop` 1.5 秒停完，不留进程、不占显存；同一个运行目录里不能起第二个引擎。
4. 这些都是热改上去的：同一个引擎从 master `438d14b` 的树冷启动，调度器的修复和 sleep 的代码都用 `apply` 加上去，中途没有重启。

## 改了什么

### 出错以后（`core.sched`）

- **出错时就结束引擎里的全部请求**（`Scheduler._fail`）。不再发步以后，在跑的请求永远结束不了，留着只会让客户端挂着、引擎空不了。现在引擎立刻是空的，后面的变更不带参数就能 apply。`drain.py` 里"出错以后只有 `--force` 能清空"的检查删了。
- **变更提交以后重新开始服务**（`Scheduler._recover`）：
  - 每一步开头比较现在服务的身份和出错时的身份，不同就说明有变更提交了（改代码、改配置，只改 live 键也算）；
  - 循环像一个新实例那样重来：出错时还在途的步交给 `_adopt` 放弃（它们的请求已经结束，占的东西还给账本），下一份计划是全量增量，worker 里留下的行一并丢掉；
  - 这些步后来才到的回答直接丢弃。
- 变更被撤回时身份不变，故障留着。变更没修好的话，下一步会同样失败。

### 让出显存（`holder.sleep`）

新配置键 `holder.sleep`，属于 `core.state.holder`，启动时绑定（改它就是一次要求引擎为空的变更）：

| 单元 | 睡着时 |
|---|---|
| `core.state.holder` | 每种状态给一项，不看预算，也不看 `state.kv.pages`、`state.gdn.slots` |
| `state.kv_paged`、`state.gdn_slots` | 按 holder 缩到一项 |
| `core.state.weights` | 声明的张量放在 meta 设备上：有形状，不占内存，不读 checkpoint；`refit` 被拒 |
| `core.plan` | 不录 graph（`plan.cuda_graph=required` 时也不算失败） |
| `core.sched` | 拒绝所有请求，回 400，说引擎睡着了；不排步 |

醒来是把它改回 false：这几个单元重载，从 checkpoint 重新读权重，重建状态池，重录 graph。和 vLLM 第 2 级的 sleep 一样不留权重副本，`refit` 换过的权重回到 checkpoint。SGLang 的 `release_memory_occupation` 用 torch_memory_saver 留住虚拟地址，graph 不用重录；我们的 graph 本来就随 `core.plan` 重载重录，用不着。

### 命令行（`deploy/agxinfer.py`、`deploy/models/`）

- 起停引擎：`serve <模型目录>`（`--detach` 在后台跑，`--dry-run` 只生成拓扑和配置），`status`，`stop`，`sleep`，`wake`。
- 改引擎：`control check|apply|status|introspect|logs`，就是控制面客户端，地址从运行目录里找。
- 发请求：`chat`、`complete`。生成测试用的 checkpoint：`checkpoint`。
- 一个运行目录（`--run`）就是一个引擎，命令都靠它找到引擎。`serve` 写一份 `engine.json`。没有 `engine.json` 的引擎（比如用旧脚本起的）也找得到：控制面地址从 `control/control.json` 读，URL 从 `config.json` 读。
- `stop` 不靠记下的 PID：按进程参数精确地找到这个运行目录的控制面（`core.control serve --state-dir <run>/control`）和它起的进程（参数里有运行目录下的路径，而且是节点代理、进程宿主或控制面），给控制面发 SIGTERM，等它们全部退出，到时限还没退的 kill 掉。
- 模型相关的东西放进 recipe（`deploy/models/qwen38.py`），按 `config.json` 选：这个家族独有的参数（`--ssm-dtype`、`--conv-dtype`）、它的单元读的键（`alloc.states`、`model.*`、`state.gdn.*`）、`checkpoint`。以后加别的模型，加一个 recipe。
- `--set 键=值`：没有专门参数的配置键。
- 和 vLLM、SGLang 对齐的名字：模型目录是位置参数（`--model` 还认），`--host`（`--bind` 还认），`--served-model-name`（`--name` 还认）。

### 临时检查

- skill 的「在引擎里做检查」改成：检查写在另一份工作区里（`rsync` 一份 `engine/`），从那里 apply，查完从仓库 apply 一次就撤掉了。这一轮就是这样做的（下面的 `ws_probe`）。
- `tests/scenarios/test_no_temporary_checks.py`：`engine/` 里有标着 `# PROBE` 的行就失败。

## 在跑着的引擎上的验证

4090 D，Qwen3.8 结构的 4 层 bf16 随机权重，`--cuda-graph auto --overlap --max-rows 8 --max-context 4096`。引擎从 master `438d14b` 导出的树（`ws_master`）冷启动，身份 `579ff21f132719ae`。每一步之后用同一个 prompt 贪心生成 24 个 token，和冷启动时的基线比。

### 出错和恢复

| 步骤 | 结果 |
|---|---|
| apply 仓库（调度器的修复） | 提交，重载 `core.sched`，身份 `8a64ef2bc55e3730`，输出和基线相同 |
| apply `ws_probe`：仓库的副本，executor 的 `submit` 里加了检查，第 3 个 decode 步抛 `RuntimeError("a bug in the probe")` | 提交。发一个请求：**0.05 秒回 400**，`the engine stopped serving at step 52: worker failed: RuntimeError: a bug in the probe`；下一个请求 400 |
| 不带参数 apply 仓库（撤掉检查） | **提交**（以前要 `--force`，而且提交了也不恢复）。身份回到 `8a64ef2bc55e3730`，输出和基线相同。调度器的日志：`stopped at step 52 ... 1 request(s) ended with it`、`serving again: a change committed after the failure at step 52`、`took over: step 52 was abandoned` |
| `holder.mem_fraction` 0.95 | 提交，显存 23,886 MiB。6 个约 2000 token 的 prompt 同时发：**全部在 0.29 秒内回 400**，`OutOfMemoryError: CUDA out of memory. Tried to allocate 20.00 MiB`，第 77 步 |
| `holder.mem_fraction` 0.97 | 放不下，变更撤回。故障留着：6 个 prompt 再发一次，全部 400 |
| `holder.mem_fraction` 0.85 | 提交，只重载了 worker 里的 `core.state.holder`，调度器没有重载。身份回到 `8a64ef2bc55e3730`，显存 21,480 MiB；6 个长 prompt 全部 200（0.36 秒），输出和基线相同 |

### 睡和醒

| 步骤 | 结果 |
|---|---|
| apply 仓库（sleep 的代码） | 提交，重载调度器和 worker 里的 `core.plan`、holder、weights、两个状态存储；身份 `b2e8f90910d65d30`，输出和基线相同，21,482 MiB |
| `holder.sleep` true | 2.05 秒提交（执行 0.09 秒），**962 MiB**。holder 的日志：`asleep (holder.sleep): one item of each of ['gdn.slots', 'kv.paged']`；`core.plan`：`"buckets": [], "reason": "asleep (holder.sleep)"`。请求回 400：`the engine is asleep (holder.sleep): it holds no weights or states...` |
| `holder.sleep` false | 3.61 秒提交（执行 1.6 秒），21,480 MiB，graph 桶 `[1, 2, 4, 8]`，输出和基线相同，6 个长 prompt 全部 200 |
| 再睡、再醒三轮 | 每轮睡着 962 MiB、醒来 21,480 MiB，输出都和基线相同 |

醒来的身份是 `f0d49620a6b173b2`，和第一次睡之前的 `b2e8f90910d65d30` 不同：`holder.sleep` 这时显式写进了配置。之后每次醒来都是 `f0d49620a6b173b2`。

### 命令行

- 对上面那个引擎（用旧脚本起的，没有 `engine.json`）：`status`、`complete`、`sleep`、`wake`、`control status` 都找得到它。`stop` 1.5 秒停完，显存回到 1 MiB，没有留下进程；再 `stop` 一次说没有引擎，退出码 0。
- `serve <checkpoint> --detach`：16 秒后引擎起来、命令返回。再往同一个运行目录 `serve` 被拒，退出码 1。`chat` 能用。
- **边开着引擎边跑 GPU 测试**：这个引擎 `sleep`（962 MiB），然后在同一张卡上跑 `tests/gpu/` 的 8 个文件，全部通过：

  | 文件 | 结果 |
  |---|---|
  | `test_real_kernels.py` | 26 passed |
  | `test_qwen38_partial.py` | 4 passed |
  | `test_cuda_graph.py` | 9 passed |
  | `test_cuda_graph_adversarial.py` | 3 passed |
  | `test_state_rebuild.py` | 1 passed |
  | `test_cuda_graph_apply.py` | 4 passed（自己按 0.85 起引擎） |
  | `test_qwen38_overlap.py` | 1 passed |
  | `test_qwen38_serve.py` | 2 passed |

  跑完 `wake`，3.36 秒，输出和睡之前逐字相同。这些测试起引擎走的也是新的命令行。

## 测试

都进 CI：

- `tests/units/scheduler/test_scheduling.py`：
  - 出错时在跑的请求立刻回错误；
  - 出错以后变更提交，服务恢复，第一份计划是全量增量，开不开重叠都测；
  - 没修好的变更会再出错，再下一次变更又恢复；
  - 重载调度器以后照常服务。
  - 开重叠时，出错那一步之后的一步已经在路上了：夹具 worker 收到一份以为出错那步跑过的计划，会报错。测试只要求变更之后不再有这种报错。
- `tests/units/scheduler/test_drain.py`：出错以后 refuse、wait、force 都直接排空；等待中的变更碰上 worker 出错，照常提交，服务恢复。
- `tests/units/worker/test_worker.py`：真 worker 栈上 executor 抛异常，变更提交后生成和 HF 逐 token 相同，开不开重叠都测。
- `tests/scenarios/test_m2a.py`：CPU 上用命令行在后台起引擎，然后 `status`、`sleep`（请求被拒）、`wake`（输出和睡前相同）、`control check`、`complete`、`stop`（不留进程）；同一个运行目录里第二个引擎被拒。
- `tests/scenarios/test_no_temporary_checks.py`。

## 没做的、要注意的

- **醒的时候读权重在暂停期里**，受执行时限（`supervisor.rpc_timeout_s`，30 秒）约束。4 层 7.6 GiB 用 1.6 秒；完整 64 层模型上没试过，可能超时，超时的话变更撤回，引擎还睡着。第二个 PR 让单元能在准备阶段做慢的事，以后可以把读 checkpoint 也挪过去。
- 睡着时剩下的 962 MiB 是 CUDA 上下文和 kernel 库自己的工作区（FlashInfer 的 384 MiB 等），没有放。
- apply 救不回来的仍然救不回来：CUDA 上下文坏了（非法访存、device 端 assert）、进程死了、某一步卡住。这些要重启 worker 进程，现在没有不重启引擎的办法。
- 只改状态内容、不换身份的变更（`refit`）不会让调度器恢复。
- 第一次醒来的身份和睡之前不同（见上）。
- `stop` 在 Linux 上读 `/proc`；别的系统用 `ps` 的输出，路径里有空格时可能找不全。
