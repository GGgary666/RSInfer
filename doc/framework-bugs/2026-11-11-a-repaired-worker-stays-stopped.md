# worker 出错以后，把它改好的变更提交了，引擎却继续拒绝请求

- 日期：2026-11-11
- 报告人：Claude Fable 5.2（在跑着的引擎里临时加检查代码时发现，见 [../rsi/2026-11-11-real-kernels-hot-apply.md](../rsi/2026-11-11-real-kernels-hot-apply.md)）
- 状态：已修（2026-11-11，见文末「修复」），修在 `core.sched`，单元一侧，热改上去就生效。属于"单元为了热更新实现的东西"：调度器（`core.sched`）在变更提交以后不清掉出错状态。变更协议本身没有问题。
- 引擎：RTX 4090 D，4 层 bf16 的 Qwen3.8。`deploy/qwen38.py serve --model <checkpoint> --tokenizer <tokenizer> --cuda-graph auto --overlap --max-rows 8 --max-context 4096`，worker 在 `cuda_graph` 平台。变更前的身份 `622273f995128992`。
- 变更：
  1. 在 `engine/executors/pytorch/unit.py` 的 `submit` 开头加几行：第 3 个 decode 步时 `raise RuntimeError("a bug in the probe")`。`apply --source <工作区>`，提交（这种只在处理请求时才出错的改动，apply 发现不了）。
  2. 发一个请求。worker 的日志：`core.worker: step 230 failed: RuntimeError: a bug in the probe`。
  3. 把文件改回去，`apply --source <工作区>`，不带参数。提交，重载 `executors.pytorch`，身份回到 `622273f995128992`。
- 现象：
  - 第 3 步以后，每个请求仍然回 HTTP 400：`the engine stopped serving at step 230: worker failed: RuntimeError: a bug in the probe`。worker 这时跑的已经是好的代码。
  - 接着试的几种变更都提交了，都没有恢复服务：

    | 变更 | 重载了什么 | 之后的请求 |
    |---|---|---|
    | `apply --source <工作区> --force`（没有任何改动） | 无 | 400 |
    | `apply --config '{"plan.cuda_graph": "off"}' --force` | `core.plan` | 400 |
    | 在 `engine/core/sched/loop.py` 末尾加一行注释，`apply --source` | `core.sched` | 200 |

  - 出错那一刻在途的那个非流式请求一直没有返回，我的客户端等了 120 秒超时。客户端断开以后引擎里没有请求了，所以第 3 步不带参数的 apply 没有被拒。
- 期望：把 worker 改好的变更提交以后，引擎恢复服务；至少 apply 的结果里要说明引擎还停着、怎样恢复。依据：
  - 调度器出错时自己的日志写的是 `New requests get an error; the requests held stay until the engine is changed`（`core/sched/loop.py`），意思是变更就是出路；
  - `doc/specs/empty-apply.md` §5 只说出错以后要用 `--force` 才能清空，没有说清空并变更以后出错状态怎么办；
  - skill 的"可能是更新机制的 bug"里有"提交以后……引擎停止服务"这一条。
- 原因：`core/sched/loop.py` 的 `self.fault` 只在 `__init__` 里置空、在 worker 报失败时赋值，没有任何地方清掉它。`core/sched/drain.py` 的 `commit()` 只重新向每个 rank 要 `Capacity`。所以只有调度器这个单元自己被重载（换一个新的 loop 实例）时才恢复。
- 复现：上面"变更"的三步。能写成测试：`tests/units/scheduler/test_drain.py`，接在 `test_after_a_worker_failure_only_force_empties_the_engine_and_the_change_says_so` 后面——夹具 worker 按脚本失败，`force` 排空，`commit()`，换新身份并报 `Capacity`，再发一个请求，期望它被服务。
- 绕过：让 `core.sched` 重载一次。在 `engine/core/sched/` 的某个文件末尾加一行注释，apply；再把注释去掉，apply。这次就是这样做的，之后身份和输出都回到原值，没有重启引擎。重启引擎也可以。
- 修复（2026-11-11，用户同意了讨论里的两个方向）：
  - **出错时就结束引擎里的全部请求**（`Scheduler._fail`）。原来在跑的请求留着，但不再发步，它们永远结束不了：客户端挂着，引擎空不了，只能 `--force`。现在出错那一刻它们就回错误，引擎是空的，后面的变更不带参数就能 apply。
  - **变更提交以后重新开始服务**（`Scheduler._recover`）。每一步开头比较现在服务的身份和出错时的身份，不同就说明有变更提交了，包括只改 live 键的变更。这时循环像一个新实例那样重来：
    - 出错时还在途的步交给 `_adopt` 放弃，它们的请求已经结束，占的东西还给账本；
    - 下一份计划是全量增量，worker 里留下的行一并丢掉；
    - 这些步后来才到的回答直接丢弃。
  - 变更被撤回时身份不变，故障留着。变更没修好的话，下一步会同样失败。
  - `drain.py` 里"出错以后只有 `--force` 能清空"的检查删掉了：出错以后引擎总是空的。
  - 测试（都进 CI）：
    - `tests/units/scheduler/test_scheduling.py`：出错时在跑的请求立刻回错误；出错以后变更提交，服务恢复，第一份计划是全量增量，开不开重叠都测；没修好的变更会再出错，再下一次变更又恢复；
    - `tests/units/scheduler/test_drain.py`：出错以后 refuse、wait、force 三种模式都直接排空；等待中的变更碰上 worker 出错，照常提交并恢复服务；
    - `tests/units/worker/test_worker.py`：真 worker 栈上 executor 抛异常，变更提交后生成和 HF 逐 token 相同，开不开重叠都测。
  - 在跑着的引擎上的验证（4090 D，从 master `438d14b` 的树冷启动，把修复热改上去）：
    - 照上面"变更"的第 1 步加检查：出错那一刻在途的请求 0.05 秒就回了 400，下一个请求 400；
    - 不带参数 apply 改好的树：提交，身份回到加检查之前的值，输出和基线逐字相同；
    - 真的 CUDA OOM：`holder.mem_fraction` 调到 0.95，6 个 2000 token 的 prompt 同时发，全部在 0.29 秒内回 OOM 的错误；调到 0.97 的变更被撤回（放不下），故障留着；改回 0.85，提交，只重载了 worker 里的 `core.state.holder`，调度器没有重载，6 个长 prompt 全部 200，输出和基线相同。
    - 数字和过程见 [../evaluations/recovery-sleep-cli.md](../evaluations/recovery-sleep-cli.md)。
  - apply 救不回来的仍然救不回来：CUDA 上下文坏了（非法访存、device 端 assert）、进程死了、某一步卡住。这些要重启 worker 进程，现在没有不重启引擎的办法。

