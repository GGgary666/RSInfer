# 分布式事务的有限状态验证

基于 `master@820d275`，包含 framework-base 以及 PR #10、#11 的修复。所有实现、文档和生成物均限定在本目录，不修改生产代码、项目依赖配置或既有 CI。

这里实现的是 **TLA+ 有限模型检查 + 真实 Python 实现的轨迹回放/契约测试**。它不是整份 Python 程序的形式证明，也不把单机故障注入当作 Linux 多机网络分区实验。

## 运行

需要 Java 11+、Python 3.12+ 和仓库现有开发依赖。在仓库根目录执行：

```sh
uv sync --python 3.12
uv run python formal-tests/run.py --bootstrap --replay
```

首次 `--bootstrap` 从官方 release 下载 TLC 1.7.4，校验 `run.py` 中固定的 SHA-256，缓存于 `.tools/`。后续可以省略 `--bootstrap`，或通过 `--jar /path/to/tla2tools.jar` 提供相同版本的工具。Java 路径可以通过 `--java /path/to/java` 或 `JAVA` 环境变量指定。不会自动修改系统 Java。

```sh
# 单独检查某组配置
uv run python formal-tests/run.py --case transaction-rollback
# 已生成全部轨迹后，单独运行实现检查
uv run pytest formal-tests/tests -q
# 本目录不在既有 pyright include 中，需要显式检查
uv run pyright formal-tests
```

`--timeout` 是每组模型的执行时间上限，默认 180 秒。超过上限表示验证未完成，命令返回失败。TLC 的语法错误、Java 失败、非预期不变量违反、缺少轨迹以及不匹配的工具校验和同样返回失败。没有 xfail 或缺失工具时自动 skip 的路径。

`.runs/<case>/` 保存使用的模型、生成的 `.cfg`、完整 `tlc.log`、反例/见证 `trace.json` 和 `result.json`；整次模型检查成功后产生 `.runs/report.json`。这些文件均被本目录 `.gitignore` 排除。TLC 1.7.4 的文本轨迹由 runner 转成 JSON：标量和事件字段供回放，原始完整状态仍保留，不冒充通用 TLA+ 值解析器。回放检查模型和轨迹的 SHA-256，拒绝旧模型的缓存轨迹。

## 模型和边界

| 模型 | 范围 | 主要对应代码 |
| --- | --- | --- |
| `models/Transaction.tla` | 一个事务、两个参与者、一个可移除的旧进程；请求与回执分别送达 | `Coordinator._transact/_commit/_abort/_confirm`、`Supervisor.stop`、`EventLog.decide` |
| `models/Participant.tla` | 一个参与者、两个事务 ID；重复、迟到及交错的 Prepare/Pause/Execute/Commit/Resume/Abort | `Participant.on_*`、`_undo`、`_resume`、`finished` |
| `models/SequentialTransactions.tla` | 两个参与者、两个连续事务，第一事务 Execute 后 Abort；旧请求/回执保留，成功回滚后允许替换进程 | 协调者排队/冻结、参与者结束记录、事务与回执匹配 |
| `models/Incarnation.tla` | 一个 ProcId、两个 incarnation、两个重叠 Spawn；旧 Stop/Exited/Spawn | NodeAgent 的 lifecycle lock、Spawn/Stop 检查、Supervisor 报告过滤 |

连续事务模型已经联合检查两个参与者上的两个事务，但范围明确限定为“第一事务执行后回滚，第二事务尝试提交”，包含回滚失败导致禁止第二事务、成功回滚后替换部分参与者的分支。它不是任意连续事务的证明，也尚未证明四个模型与 Python 实现之间的全量精化关系。

关键建模选择：

- `label/identity` 与实际生效的 `effective/value` 分开。不能靠返回旧身份来证明回滚成功。
- 旧进程实际存活与停止确认分开。LOST 是故障检测结果，不是停止证据。
- 持久化 Commit、发送请求、参与者执行、回执被观察是独立转换，能检查发送先于持久化以及提前 Resume 的错误。
- 已发送请求和产生的回执可任意重放；永久不送达表示丢失。该抽象不限制网络重复次数，但合并了没有可观察状态差异的重复处理。
- 底层动作以有限结果抽象：执行成功/失败、撤销成功/失败。模型不验证任意用户 hook 的内部实现。真实实现测试用可观察的 toy unit 状态检验协议对这些结果的处理。
- 事务模型允许在决定前失败，决定后确认超时并冻结。执行后未获决定时保持暂停由正常模型的停留行为表示；错误变体额外允许擅自回滚。
- 替换进程会清空其内存中的 finished/tombstone；模型没有假定这些记录可以自动恢复。连续事务模型把已发请求绑定到原 endpoint generation。**这是一条传输隔离前提，不是 Python change 消息已有 incarnation 字段。** `RouteAcrossIncarnation` 变体放开该前提后，可以让旧 Prepare 重新激活已结束事务；因此不能据此声称已经证明 TCP 地址复用或跨重连投递安全。
- 生命周期模型把 Spawn 的检查与异步创建完成拆开，允许两个请求同时存在。正确配置依靠同一 ProcId 的锁；移除锁必须产生同一 ProcId 多个活进程的反例。

明确未验证：协调者崩溃恢复/日志损坏、任意阶段的进程崩溃和在线恢复、地址复用下旧事务消息的隔离、拓扑恢复的所有细节、分片 Prepare 的内部资源管理、配置/reload 的全部实现、KV cache/GPU 内核和请求级数据一致性。稳定存储成功、非 Byzantine 参与者以及模型转换的原子性是当前前提。

这里只检查安全性。`CHECK_DEADLOCK FALSE` 是因为终止状态和网络永久不交付是允许的，不能把这种停留误报成协议 bug。没有加入公平性或时钟进展假设，因此没有证明最终成功、恢复服务或实际超时上界。模型规模增加需要重新检查，不能由两个参与者推广到任意数量。

## 检查的性质

事务模型检查：类型约束、先持久化再发送 Commit、决定 Commit 与参与者状态相容、恢复服务前的全体 Commit 确认屏障、停止确认的真实性、回滚失败保持暂停、失败结果要求冻结，以及 committed/aborted 结果与实际状态一致。

参与者模型检查：已结束事务不能重新激活、回滚失败保持暂停、服务状态与实际效果一致、提交的身份确实对应已执行效果、每个事务最多执行一次和撤销一次。

连续事务模型额外检查旧回执不能计入新事务、旧 Abort 不能清空新事务、失败冻结阻止第二事务、替换后不能重新激活第一事务。生命周期模型检查单个 ProcId 最多一个活进程、活进程必须可追踪、旧停止/退出/启动不能影响后继实例。

runner 内置三类配置，共 43 组：

1. **12 组正常模型**：原有 8 组，加连续事务、回滚失败、回滚后替换进程、incarnation 生命周期。必须完成状态穷举且没有违反不变量。
2. **17 个错误变体**：原有 9 类，加旧 Abort 影响新事务、旧回执混入新事务、忽略恢复冻结、跨 incarnation 投递旧 Prepare、去掉 lifecycle lock、忽略 Stop generation、接受旧报告、允许倒退 Spawn。每个变体必须违反指定不变量；任意报错不能算“抓到 bug”。其中跨 incarnation 投递刻意放宽的是传输前提，不能直接算成已确认的生产 bug。
3. **14 组可达性见证**：原有 8 组，加第二事务提交、旧消息交错后提交、回滚失败、替换后提交、后继进程运行、后继进程收到旧 Stop 后继续运行。这用于排除模型把正常路径全锁死而安全性质空泛通过的情况。见证的存在不代表活性证明。

模型错误变体等价于协议层的 C；正常模型是候选 B。额外有 3 个真实 Python 代码的内存变异验证，但没有因此声称整份生产代码的 A/B/C 或精化证明已完成。

## 和真实实现连接

`tests/test_trace_replay.py` 自动读取 TLC 生成的 5 条参与者见证和 3 条参与者错误变体轨迹，逐步调用真实 ProcessHost/Participant handler。见证要求每一步的事务 ID、阶段、实际值、身份、暂停状态和撤销失败标志与模型一致；错误变体要求真实实现拒绝遵循错误轨迹，并始终满足独立安全断言。轨迹未生成时失败，不跳过。

`tests/test_coordinator_contract.py` 在真实 Coordinator/Participant 的通信边界注入 Commit/Resume 请求或回执丢失，覆盖一次丢失与持续丢失，并检查日志决定顺序、恢复服务屏障、冻结后续变更、LOST 停止和回滚失败。它是模型性质的实现契约测试，尚不是事务模型全量轨迹精化检查。

`tests/test_checker_failures.py` 检查验证工具本身不会把语法错误、其他不变量失败、超时、缓存旧结果或错误工具当成预期反例。

`tests/test_sequential_transactions.py` 的 24 个交错用例在第二事务 Prepare/Execute/Commit/Resume 边界重发第一事务实际发送过的消息，断言活跃事务、暂停状态、实际值和 hook 次数不变。另一个用例在确认回滚后替换真实 ProcessHost 对象，并在第二事务执行时向存活参与者重发旧 Abort、向 Supervisor 发送被替换进程的旧报告。它没有假装旧事务请求可以被传输层重定向到新 endpoint。

`tests/test_incarnation.py` 使用真实 NodeAgent dispatcher/锁及 Supervisor 报告处理，只在操作系统子进程创建边界使用可控替身。故意让第一个 Spawn 悬停在真实 await 边界，再提交重复 Spawn、旧 Stop 和旧 Spawn。6 个报告用例覆盖 SPAWNING/RUNNING 下的旧 Exited/Heartbeat/CheckIn，且含当前 generation 的正对照。3 个内存变异用例临时移除 lifecycle lock、Stop generation 或 Exited generation 检查，要求同一套测试捕获错误；不会修改磁盘上的生产代码。

现有仓库 pytest/CI 不会自动发现 `formal-tests/`。需要在 Linux 验证任务中显式调用上面的完整命令；本次没有改动 `.github/`。此目录可以单独运行和审阅。

## 已执行的验证

在 macOS 上使用 TLC 1.7.4、Temurin Java 21 和 Python 3.12 运行完整入口：

- 完整入口的 43 组配置全部达到预期：12 组正常模型完成穷举，累计检查 4,743 个不同状态（按配置求和，不代表合并后的全局状态空间）。完整结果见运行生成的 `.runs/report.json`。
- 17 个错误/前提变体全部触发指定不变量，14 个可达性目标全部生成见证。
- 59 项 Python 测试通过：初版 24 项，加 25 项连续事务/替换组合，以及 10 项 incarnation/内存变异检查。
- `pyright formal-tests`：0 errors。

没有为这些测试增加 macOS 兼容分支；尚未运行本分支的 Linux CI。

## 参考

- [TLC 官方实现和发行包](https://github.com/tlaplus/tlaplus)
- [TLC 配置文件](https://docs.tlapl.us/using:tlc:config_file)
- [Gray / Lamport：Consensus on Transaction Commit](https://www.microsoft.com/en-us/research/publication/consensus-on-transaction-commit/)
