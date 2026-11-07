# framework-base：从测试集合出发的对抗验证

## 范围与结论

审查基线为 `framework-base@010c075`，测试分支为 `gw/framework-base/adversarial-tests`。本轮只增加测试及本文，不修改生产代码、既有测试、规格、HTML 或 CI，也不重复实现已合并的修复。

原集合共收集到 **166 个用例**。本轮新增 **14 个用例：8 个失败、6 个通过**，失败对应 **4 个不同问题**。断言检查的是规格要求的正确结果；没有 xfail、skip，也没有把当前错误行为写成期望值。

项目只支持 Linux。用户允许先在本机 macOS / CPython 3.12.13 上测试。本轮新增测试没有 macOS 兼容代码，也不经过快照 chmod/rename 边界；使用真实 Participant、UnitRuntime、ConfigStore、EventLog 和文件，其中 3 个用例经过真实 ZMQ RPC。Linux 上的最终复验尚未执行，不能把本机结果称为 Linux CI 结果。

## 现有集合的覆盖结构

以下数量是 pytest 展开参数后的用例数，不是代码覆盖率；Hypothesis 在单个用例内部生成的多个 example 不另计。

| 范围 | 用例数 | 现有主要验证 | 本轮关注的缺口 |
| --- | ---: | --- | --- |
| change | 72 | Commit/Resume 丢包、确认屏障、超时、常规回滚、拓扑恢复、分片准备、队列和 blob 保留 | 终止结果与迟到消息的交错；失败后再试；记录元数据与实际内容是否一致 |
| unit_runtime | 37 | 重载故障点、依赖传播、注册撤销、过期代理、staging；随机依赖 DAG | 声明阶段部分成功后失败；拒绝之后能否处理另一个候选 |
| ipc | 28 | RPC 超时、SHM wrap、有序和背压、断连、开放记录 | 传输成功不等于业务状态机对迟到消息安全；要跨真实 RPC 重放业务序列 |
| cluster | 11 | 真子进程启动、快照、进程丢失、拓扑，以及 Spawn/Stop 重复和迟到消息 | Spawn/Stop 的幂等性测试不能代替 Prepare/Pause/Execute/Abort 的时序测试 |
| context | 10 | 配置类型、所有权、读取权限、hooks、资源租约 | 单个错误值拒绝之外，多键声明的部分写入与事务撤回 |
| platforms | 7 | FakePlatform 的地址、形状、内容依赖和 graph 失效 | 只证明 fake backend 的契约，不证明真实 GPU/model 正确性 |
| location transparency | 1 | 源码中不存在约定禁止的本地地址形式 | 静态文本检查，不是跨机故障模型验证 |
| 合计 | 166 | | |

已有测试并非只看状态标签。例如 commit-consistency 检查实际单元版本及中途服务状态，SHM 测试检查真实载荷，runtime 随机测试检查依赖闭包和注册数量。这些是应保留的强断言。

薄弱处主要是**组合后的状态边界**：

- `test_any_failure_rolls_back_every_participant` 在正常阶段顺序中注入一次故障，确认 Abort 一次；没有在 Execute 已经自行撤销后重放 Pause/Execute。
- `test_an_action_whose_prepare_raised_is_still_aborted` 验证失败动作得到清理；没有继续投递同一 change id 的迟到 Prepare。
- staging 和新配置测试覆盖正常新增、代码冲突及执行后回滚；没有覆盖第二个配置值非法、前一个声明已经生效的路径。
- `test_contents_live_in_blobs_and_purges_are_recorded` 覆盖 change-set blob 本身被删；没有覆盖 change-set 仍保留、但它指向的内容已经被删。

不能据此宣称现有 166 例在 Linux 全部通过且漏掉了这些问题；本轮本机有一部分集成用例在平台前置步骤被阻断。这里的覆盖缺口结论来自测试阅读和独立反例。

## 新用例与实测结果

| 维度 | 变体 | 失败 / 通过 |
| --- | --- | --- |
| Abort 后迟到 Prepare | Abort 先到；Prepare 抛错后收到 Abort，再重放 Prepare | 2 / 0 |
| Execute 自行撤销后迟到消息 | Pause → Execute，再收到最终 Abort | 1 / 0 |
| 正常终态重复消息对照 | 正常 Abort；正常 Commit/Resume，随后重放各控制消息 | 0 / 2 |
| 真实 RPC | Abort 先到；失败后重放 Pause/Execute；正常终态对照 | 2 / 1 |
| staging 校验结果 × 观察结果 | 合法/非法第二个配置值 × 配置残留/下一候选能否 Prepare | 2 / 2 |
| 审计记录与实际 blob | 删除候选内容；明确保留内容的对照 | 1 / 1 |
| 合计 | 14 例 | 8 / 6 |

对抗输入只改变消息到达顺序、重放已有消息、提供需要拒绝的配置，或触发正常保留策略。没有修改框架实现来制造失败。

## 发现 1：已撤销的 Execute 可被迟到 Pause 重新触发（建议 P1）

位置：`engine/core/change/participant.py` 的 `on_pause`（约 362 行）、`on_execute`（约 428 行）、`_undo`。

序列：

```text
Prepare → Pause → Execute 抛错并自行撤销
        → 迟到/重复 Pause → 重复 Execute → Abort
```

第一次 Execute 失败后，阶段为 `undone`，准备动作列表已经清空。迟到 Pause 没有拒绝这个阶段，而是重新转到 `paused`，使 Execute 再次运行；第二次撤销已没有记录可清理。

实测：最终 Abort 返回成功、进程恢复服务、身份仍为 v1，但单元实际值为 2。计数为 `executes=2, aborts=1`，正确结果应为 `executes=1, aborts=1, value=1`。同一问题也通过真实 RPC 复现。

依据：`doc/specs/ipc.md §4` 的迟到与重复控制消息安全性，以及 `doc/specs/change.md §4.2/§7` 的失败回滚要求。

用例：`test_late_pause_cannot_rearm_an_execute_that_was_already_undone`，以及 RPC 的 `pause-after-failed-execute` 变体。

## 发现 2：staging 配置校验失败留下声明和代码来源（建议 P1）

位置：`engine/unit/runtime/runtime.py::_stage`（约 890–893 行）、`ConfigStore.declare_entries`、`Participant.on_prepare`（约 214–216 行）。

候选新增单元声明两个整数配置，传入第一个值 42、第二个值为字符串。框架正确返回 `Failed`，但在返回前已经修改了 staged root，并成功登记了第一个配置。由于此时还没有单元 slot，参与者也没有标记 `c.staged`，后续 Abort 未撤销这些部分修改。

两个独立断言分别发现：

- Abort 后仍有一个属于不存在单元的配置键，值为 42；
- 换一个合法候选再次 Prepare，得到 `unit.stage_conflict: another snapshot is already staged`。

合法配置的相同流程均通过，因此不是无条件要求所有 staging 失败。非法输入本就应被拒绝；问题是拒绝后污染运行环境，阻断后续合法操作。

依据：`doc/specs/change.md §4.2` 的失败撤回、`unit-runtime.md §6.3` 的 staging 撤销语义。

用例：`test_rejected_staging_restores_config_and_allows_a_new_candidate` 的两个非法配置变体。

## 发现 3：确认 Abort 后仍会接受该事务的迟到 Prepare（建议 P2）

位置：`Participant.on_abort`（约 522–524 行）与 `on_prepare`（约 183–184 行）。

当 Abort 对应的事务尚未进入 `self.change` 时，参与者返回 `Resumed(rolled_back=True)`，却不记录该 change id 已终止。随后同一事务的迟到 Prepare 仍会执行。Prepare 钩子抛错后也是同类路径：清理完成但没有终态记录，后续重放可以再次执行钩子。

实测：

- Abort 先到：随后 Prepare 返回成功，逻辑资源计数从 0 变成 1；
- Prepare 曾抛错：重放后 prepare/abort 次数从各 1 次变成各 2 次；
- 第一种序列通过真实 RPC 同样复现。

这不是已经合并的“同一个成功执行动作在一次回滚里被 Abort 两次”的旧问题；触发点是缺少终止记录后重新接受迟到请求。

依据：`doc/specs/ipc.md §4` 明确允许超时请求迟到，并要求接收方按 change id 安全处理。

用例：`test_acknowledged_abort_rejects_late_prepare` 两例，以及 RPC 的 `abort-before-prepare` 变体。

## 发现 4：引用内容被清理，审计记录仍称内容未清理（建议 P2）

位置：`engine/core/change/eventlog.py::prune`（约 191–193 行）。

一个已撤回变更的 base 仍是当前身份，因此其 change-set blob 被保留。候选文件内容不再受保护，被正常回收；但 `content_purged` 只检查 change-set blob 自身是否存在，没有检查它引用的文件内容。

实测：`changeset(cid)` 仍能返回元数据，`blobs.has(candidate)` 为 False，记录却为 `content_purged=False`。明确保留候选内容的对照例通过。

依据：`doc/specs/change.md §5.1` 要求内容被清理的记录标明“内容已清理”。

用例：`test_aborted_change_reports_when_referenced_content_was_pruned[pruned-candidate]`。

## 为什么这些测试不只检查报错

- 检查实际单元值、动作调用次数和资源计数，防止“返回 Failed/旧身份但继续修改状态”逃过测试。
- 对被拒绝的 Prepare 继续执行 Abort，再尝试合法的新候选，验证后续可用性。
- 配置残留与下一候选被阻断分别成例，避免第一处失败遮住第二个后果。
- RPC 用例经过真实序列化、ZMQ 收发、ProcessHost.safe_point 和真实 Deferred；直接调用用例只替换 Deferred 回执容器。
- 保留 6 个正向对照，约束正常终态重放、合法 staging 以及未清理内容的记录语义，防止“一律拒绝”或“一律标已清理”的伪修复。
- 本轮未实现修复 B，也未构造新的生产对抗补丁 C；这是对当前实现 A 的反例集。后续逐项修复时再继续 A/B/C 测试加强，不能把这轮称为已完成的 A/B/C 修复实验。

## 复现与环境边界

在此分支根目录、安装项目依赖后执行：

```sh
uv sync --python 3.12
uv run pytest tests/framework/adversarial -q --tb=short
uv run pyright
```

可单独运行真实 RPC 用例：

```sh
uv run pytest tests/framework/adversarial/test_transaction_boundaries.py -q -k cross_real_rpc
```

本机结果：

| 集合 | 通过 | 失败 |
| --- | ---: | ---: |
| 原有 166 例 | 113 | 53 |
| 新增 14 例 | 6 | 8 |
| 合并收集的 180 例 | 119 | 61 |

原有 53 个失败的名称在合并运行前后完全相同：51 个被 macOS 快照 rename 权限阻断，另有 TCP 背压断言、`typing.io.__spec__` 各 1 个。按 Linux-only 项目约定，这些不计为本轮产品缺陷，也未进行兼容修复。原来通过的 113 例没有新增回归。

新增文件的 Pyright 检查通过。新用例中的 8 个失败都是功能断言失败，没有导入错误、接口不匹配或平台异常。两条 RPC 反例证明故障经过真实消息边界仍存在，但单进程内驱动的 RPC 服务不等于多机网络故障验证；Linux、实际跨机、GPU/模型以及分布式随机调度仍需后续验收。
