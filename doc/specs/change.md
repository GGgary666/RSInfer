# 规格：变更事务 `core.change`

- 类型：框架本体。跑在控制面进程。改了它要重启引擎。v0 用 Python；它只通过版本化消息和其他进程交互，以后可以换成 Rust。
- 公共约定见 [README.md](README.md)。

## 1. 职责

1. 接收 change-set：来自 agent 的 `apply`、配置修改、单元自己发起的内部变更（例如 EPLB）、启动（从空系统到某个身份）。
2. 按固定流程推进：静态检查 → 推导动作计划 → 准备 → epoch 边界暂停 → 执行 → 提交或回滚 → 恢复。
3. 作为两阶段提交的协调者。
4. 返回诚实的结果，并写事件日志。

## 2. 不负责

- 不判断要动哪些东西（问 `core.deps`）。
- 不执行具体动作（重载、重录 graph、refit、搬专家……都由参与进程里的单元执行）。
- 不做鉴权、不做策略钩子（v0 不做任何安全和权限）。

## 3. change-set

```python
class ChangeSet(msgspec.Struct):
    base: Identity                       # 起点身份
    files: dict[str, str | None]         # 路径 -> 新内容的 hash；None 表示删除
    config: dict[str, Any]               # 配置键 -> 新值
    deps: dict[str, str]                 # 依赖锁的变化
    topology: Topology | None            # 拓扑数据的变化（见 supervisor.md）
    artifacts: dict[str, str]            # 名称 -> 内容 hash
    source: Literal["apply", "config", "internal", "startup", "operator"]
    meta: ChangeMeta                     # 提交者（自报）、意图、证据链接
```

- 文件内容按 hash 存在内容寻址的 blob 存储里；unified diff 只用于展示。
- 候选身份 = f(起点身份, change-set)，必须是确定的。

## 4. 流程

### 4.1 状态机

```
RECEIVED → CHECKED → PLANNED → PREPARING → PREPARED → PAUSING → EXECUTING → COMMITTED
                │          │           │            │            │
                └──────────┴───────────┴────────────┴────────────┴──▶ ABORTING → ABORTED
```

同一时刻只处理一个 change-set，其余排队。排队规则见 4.5。

### 4.2 各阶段

1. **RECEIVED → CHECKED**：起点身份必须等于当前运行身份（内部变更除外，它以当前身份为起点）；调用 `core.manifest` 做静态检查；不通过就 `Failed`，什么都不动。
2. **CHECKED → PLANNED**：调用 `core.deps` 得到动作计划：每个参与进程要做什么、哪些动作会影响在途请求、哪些要重启进程或引擎。带 `--dry-run` 时到这里就返回计划。
3. **PLANNED → PREPARING**：让每台机器的节点代理拿到新内容并物化快照目录；向每个参与进程发 `Prepare`。参与进程做能在后台做的事（装新增单元、后台重录 graph），回 `Prepared` 或 `Failed`。
4. **PREPARED → PAUSING**：按需排空受影响的请求（超时默认 10 秒，超时后中止剩余的受影响请求）；向所有参与进程发 `Pause(epoch=E)`，要求在 epoch E 结束后停下。
5. **PAUSING → EXECUTING**：所有参与进程都停下后，发 `Execute`：执行推迟的动作（重载修改过的单元、暂停期内重建派生产物、状态动作）。需要重启进程的，交给 `core.supervisor`。
6. **EXECUTING → COMMITTED**：所有参与进程回 `Executed` 后，发 `Commit`，切换到新身份，恢复服务。
7. **任一步失败 → ABORTING**：向已经准备或执行过的参与进程发 `Abort`，它们各自回滚（单元运行时的回滚见 [unit-runtime.md](unit-runtime.md) 6.2），然后恢复服务；身份不变。

### 4.3 协议消息

| 消息 | 方向 | 内容 |
|---|---|---|
| `Prepare` | 协调者 → 参与者 | change-set id、给这个进程的动作片段、新快照的身份 |
| `Prepared` / `Failed` | 参与者 → 协调者 | 诊断 |
| `Pause` | 协调者 → 参与者 | 在哪个 epoch 之后停下 |
| `Paused` | 参与者 → 协调者 | 实际停在的 epoch |
| `Execute` / `Executed` | 双向 | 动作执行结果 |
| `Commit` / `Abort` | 协调者 → 参与者 | — |
| `Resumed` | 参与者 → 协调者 | — |

- 参与者按 `ProcId` 寻址，可以跨机（P7）。
- 每条消息都带 change-set id，参与者对重复消息幂等。
- 参与者在 `Prepared` 之后、收到 `Commit` 或 `Abort` 之前失去协调者：等待一个超时（默认 60 秒，配置项 `change.coordinator_timeout_s`）后自行回滚并恢复服务，报告诊断。

### 4.4 失败的边界

- 控制面进程崩溃 = 引擎重启。事件日志在磁盘上，不丢。
- 某个参与进程崩溃：整体 abort；由 `core.supervisor` 按拓扑决定要不要重建它。

### 4.5 排队规则（已确认）

1. **不抢占**：正在走流程的 change-set 不会被打断，后来的都排队。
2. **优先级**：故障处理（进程丢失、机器丢失）> 内部变更 > agent 的 apply 和其他来源。内部变更优先，是因为它短（例如 EPLB 重排不重录、不排空），而且时效性强（负载统计放久了就过时）。
3. **防饿死**：有 apply 在等时，内部变更最多连续插一次队。
4. **内部变更可以丢弃，apply 永远不丢**：内部变更带着“在哪个身份下算出来的”（`computed_under`）。轮到它执行时，如果当前身份已经不是 `computed_under`，就直接丢弃并记一条事件；发起它的单元下个周期重新算。apply 代表 agent 的意图，只会成功、失败或被拒，不会被静默丢弃。

## 5. 结果

```python
class ChangeResult(msgspec.Struct):
    status: Literal["committed", "aborted", "failed", "planned"]  # planned 只用于 dry-run
    old_identity: Identity
    new_identity: Identity | None
    plan: ActionPlan             # 实际执行的动作计划
    timings: dict[str, float]    # 各阶段耗时，包括排空耗时
    aborted_requests: int        # 因版本切换中止的请求数
    restarted: list[ProcId]
    diagnostics: list[Diagnostic]
```

结果同时写入事件日志，见 5.1。

### 5.1 事件日志和 blob 存储（已确认）

sqlite 只存索引和摘要，不存 patch 的内容：

| 放哪里 | 存什么 |
|---|---|
| **blob 存储**（控制面本机目录，按内容 hash 寻址） | 文件内容；change-set 本身（“路径 → 内容 hash”的清单）；每个身份的完整文件清单；大的产物 |
| **sqlite 事件日志**（控制面本机） | 每条变更的编号、来源、状态、起点和新身份、状态版本、change-set 的 hash（指向 blob）、配置的改动、实际执行的动作计划、各阶段耗时、诊断 |
| **按需生成，不存** | 给人审查的 unified diff：从两个身份的 blob 现算 |

- 同一份文件内容不管出现在多少个 change-set 里只存一份；节点代理物化快照时也从这个 blob 存储按 hash 拉文件，不另存一份。
- blob 按保留策略清理（保留当前身份、正在准备的身份、最近若干个身份引用的 blob；保留数是配置项 `change.blob_retention`，默认 20）。事件记录本身永远保留；内容被清理的记录标明“内容已清理”。
- 引擎不承担完整历史的永久保存；需要的话由 RSI 平面的谱系（bare git 仓库）负责。

## 6. 内部变更入口

单元（例如调度宿主里的 EPLB 策略）通过控制面提供的服务提交 change-set，`source = "internal"`。走同一条流程（epoch 边界、两阶段提交），区别有三点（已确认）：

1. **只改状态内容的内部变更不产生新身份。** 例如 EPLB 重排只改专家放置表和专家槽位里的内容，这和 KV cache 里存了什么一样，属于状态内容，不属于“系统是什么”（参照模式下换放置位置，输出不变）。事件日志里记成“状态版本 + 1”，身份不变。这样它不会让排在后面的 apply 因为起点对不上而被拒。
2. 内部变更必须带 `computed_under`（算出它时的身份），过时就丢弃（4.5 第 4 条）。
3. 会改变身份的内部变更（如果以后有）照常产生新身份，和 apply 一样处理。

## 7. 验收测试（阶段 1）

1. 每个场景（modules.html 的 9 个）走完整流程，结果和事件日志正确。
2. 原子性：在任一阶段、任一参与者注入失败，所有参与者回滚，身份不变，服务继续。
3. 失去协调者：参与者超时后自行回滚。
4. dry-run 的计划和实际执行的计划完全一致。
5. 排空：不影响在途请求的改动不排空；超时后剩余请求以“因版本切换中止”结束。
6. 一步之内只用一个版本：暂停前后的每一步，所有参与进程报告的身份一致。
7. 两个假主机：参与者分在两台“主机”上，以上全部成立。
8. 排队：同时到达的故障处理、内部变更、apply 按优先级执行；有 apply 在等时内部变更最多连续插一次队；正在走流程的变更不被打断。
9. 过时丢弃：apply 先提交改变了身份后，排在后面、`computed_under` 是旧身份的内部变更被丢弃并记事件；apply 永远不被丢弃。
10. 身份中立：只改状态内容的内部变更提交后，身份不变、状态版本加 1，排在后面的 apply 不会因为起点不符被拒。
11. 事件日志：sqlite 里没有文件内容；从事件记录和 blob 存储能重建 change-set 和 diff；blob 被清理后记录标明“内容已清理”。
