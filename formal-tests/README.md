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

两份模型分别检查协调者和参与者的义务，尚未做它们的组合精化证明；不等于“两个参与者上的两个连续事务已经被联合穷举”。

关键建模选择：

- `label/identity` 与实际生效的 `effective/value` 分开。不能靠返回旧身份来证明回滚成功。
- 旧进程实际存活与停止确认分开。LOST 是故障检测结果，不是停止证据。
- 持久化 Commit、发送请求、参与者执行、回执被观察是独立转换，能检查发送先于持久化以及提前 Resume 的错误。
- 已发送请求和产生的回执可任意重放；永久不送达表示丢失。该抽象不限制网络重复次数，但合并了没有可观察状态差异的重复处理。
- 底层动作以有限结果抽象：执行成功/失败、撤销成功/失败。模型不验证任意用户 hook 的内部实现。真实实现测试用可观察的 toy unit 状态检验协议对这些结果的处理。
- 事务模型允许在决定前失败，决定后确认超时并冻结。执行后未获决定时保持暂停由正常模型的停留行为表示；错误变体额外允许擅自回滚。

明确未验证：协调者崩溃恢复/日志损坏、进程 incarnation 和替换协议、拓扑恢复的所有细节、分片 Prepare 的内部资源管理、配置/reload 的全部实现、KV cache/GPU 内核和请求级数据一致性。稳定存储成功、非 Byzantine 参与者以及模型转换的原子性是当前前提。

这里只检查安全性。`CHECK_DEADLOCK FALSE` 是因为终止状态和网络永久不交付是允许的，不能把这种停留误报成协议 bug。没有加入公平性或时钟进展假设，因此没有证明最终成功、恢复服务或实际超时上界。模型规模增加需要重新检查，不能由两个参与者推广到任意数量。

## 检查的性质

事务模型检查：类型约束、先持久化再发送 Commit、决定 Commit 与参与者状态相容、恢复服务前的全体 Commit 确认屏障、停止确认的真实性、回滚失败保持暂停、失败结果要求冻结，以及 committed/aborted 结果与实际状态一致。

参与者模型检查：已结束事务不能重新激活、回滚失败保持暂停、服务状态与实际效果一致、提交的身份确实对应已执行效果、每个事务最多执行一次和撤销一次。

runner 内置三类配置，共 25 组：

1. **8 组正常模型**：正常/停止/LOST/回滚失败，以及参与者执行与撤销失败的四种组合。必须完成状态穷举且没有违反不变量。
2. **9 个错误变体**：提前服务、LOST 当作停止、吞掉回滚失败、不冻结、未持久化就发送 Commit、擅自回滚、遗忘 Abort、重新激活已撤销 Execute、撤销失败后解除暂停。每个变体必须违反指定不变量；任意报错不能算“抓到 bug”。
3. **8 组可达性见证**：用刻意否定“能提交/能回滚/能进入恢复”等目标状态的不变量，让 TLC 找到到达它们的路径。这用于排除模型把正常路径全锁死而安全性质空泛通过的情况。见证的存在不代表活性证明。

这些错误变体等价于协议层的 C；正常模型是候选 B。它们是模型 mutation，不应当标注为生产代码 A/B/C 已完成。

## 和真实实现连接

`tests/test_trace_replay.py` 自动读取 TLC 生成的 5 条参与者见证和 3 条参与者错误变体轨迹，逐步调用真实 ProcessHost/Participant handler。见证要求每一步的事务 ID、阶段、实际值、身份、暂停状态和撤销失败标志与模型一致；错误变体要求真实实现拒绝遵循错误轨迹，并始终满足独立安全断言。轨迹未生成时失败，不跳过。

`tests/test_coordinator_contract.py` 在真实 Coordinator/Participant 的通信边界注入 Commit/Resume 请求或回执丢失，覆盖一次丢失与持续丢失，并检查日志决定顺序、恢复服务屏障、冻结后续变更、LOST 停止和回滚失败。它是模型性质的实现契约测试，尚不是事务模型全量轨迹精化检查。

`tests/test_checker_failures.py` 检查验证工具本身不会把语法错误、其他不变量失败、超时、缓存旧结果或错误工具当成预期反例。

现有仓库 pytest/CI 不会自动发现 `formal-tests/`。需要在 Linux 验证任务中显式调用上面的完整命令；本次没有改动 `.github/`。此目录可以单独运行和审阅。

## 已执行的验证

在 macOS 上使用 TLC 1.7.4、Temurin Java 21 和 Python 3.12 运行完整入口：

- 8 组正常模型完成穷举，累计检查 4,002 个不同状态（按配置求和，不代表合并后的全局状态空间）。
- 9 个错误变体全部触发指定不变量，8 个可达性目标全部生成见证。
- 24 项 Python 测试通过：8 项轨迹回放、10 项协调者契约、6 项检查器失败处理。
- `pyright formal-tests`：0 errors。

没有为这些测试增加 macOS 兼容分支；尚未运行本分支的 Linux CI。

## 参考

- [TLC 官方实现和发行包](https://github.com/tlaplus/tlaplus)
- [TLC 配置文件](https://docs.tlapl.us/using:tlc:config_file)
- [Gray / Lamport：Consensus on Transaction Commit](https://www.microsoft.com/en-us/research/publication/consensus-on-transaction-commit/)
