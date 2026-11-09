# 调度器（2026-11-08）

动机：阶段 2 第 3 步。把调度器重写成：
- 两个决策点；
- 在途步队列；
- 多 rank；
- 出错不冻结。

接在第 2 步的请求表（`sched.reqs`）和 #23 的数据通道上。账本从已关闭的 #22 搬来。worker 用测试夹具代替，真正的 `core.worker` 是第 4 步。

## 单元和接口

| 单元 | 种类 | 提供 | 说明 |
|---|---|---|---|
| `core.sched` | 逻辑 | `proc.loop@1`、`sched.status@1` | 每步收请求、排计划、发给每个 rank、收结果、发 token（`loop.py`） |
| `policies.fcfs` | 逻辑 | `policy.order@1` | 准入和排序：先来先服务 |
| `policies.prefill_first` | 逻辑 | `policy.compose@1` | 这一步怎么组成：prefill 优先，不混合 |
| `core.state.alloc` | 状态 | `ledger@1` | 联合账本（#22 的 `Pool`、`JointLedger`） |
| `sched.reqs` | 状态 | `sched.reqs@1` | 请求表（第 2 步） |

契约（`core.contracts`）：
- `ledger.py`：调度器调用账本的最小调用集；
- `policy.py`：两个决策点的快照（`Candidate`、`StepView`）和决定（`Take`）；
- `records.py`：
  - 新加 `StepFailure`（`worker.failure@1`）；
  - `Sampled` 写明了每一行都采样：prompt 还没算完的分块也采，调度器丢掉它的 token。sglang 和 mini-sglang 对分块请求都是这么做的。

## 来源

- mini-sglang `9a91cfa`（MIT，`LICENSE.mini-sglang`）：
  - 能组成 prefill batch 就跑 prefill，否则跑 decode；
  - 按顺序准入，遇到第一个放不下的就停；
  - 准入时按 prompt 加 `max_tokens` 一次预留（`PrefillAdder`、`estimated_len`）。
- sglang 的默认值：
  - `schedule_policy = fcfs`；
  - prefill 优先（`get_next_batch_to_run`）；
  - `enable_mixed_chunk = False`；
  - token 预算默认 2048，即 `chunked_prefill_size` 最小的那一档。
- 按 prompt 加 `max_tokens` 一次预留是 mini-sglang 的做法，sglang 自己不这样，这里照搬上游，没有修。

## 每步怎么排

1. 快照：每个活着的请求按到达顺序分成三类：
   - 等待：还没准入；
   - prefill 中：准入了，prompt 还没排完；
   - decode：prompt 排完了，还能再采。

   都按"已经排出去的步"来算，包括还在途的步。
2. 问 `policy.order@1` 要等待请求的尝试顺序，再问 `policy.compose@1` 这一步跑哪些请求、每个算几个 token。
3. 检查决定：
   - 请求不重复，而且这一步能跑；
   - prefill 不超过剩下的 prompt，decode 正好 1 个；
   - prompt token 总数不超过预算，新行不超过空行数。

   不合法就不执行，`report().policy_error` 写明原因；策略改好了，下一轮照常继续。
4. 准入：等待请求按决定里的顺序问账本，遇到第一个放不下的就停，后面的等待请求都不准入。
   - 永远放不下的请求（prompt 加 `max_tokens` 超过总容量）直接回错误。
   - 如果这一步因此什么都没排进去，就不带等待请求再问一次 `compose`：sglang 组不成 prefill batch 时跑 decode，这里也一样。
5. 先在请求表里记下这一步（`start_step`），再把计划放进每个 rank 的发件箱。

## 在途步和一步重叠

- 在途步是一个队列：没开重叠时最多 1 步；开了 `sched.overlap`（live 键，默认关）最多 2 步（stage2-contract.md §4.4）。
- 一行 decode，如果它上一次跑还在途，就用 `from_previous = 1`，输入 token 由 worker 在显存里接上；否则调度器自己给出 token。
- 请求可能在一个结果还没回来的步里就结束了：
  - 下一步里多算的那个 token，由请求表丢掉；
  - 长度到上限是能提前算出来的，这样的请求不再排。
- 账本在准入时已经按完整长度预留，所以提前排的那一步不需要另外分页。
- 请求结束后，条目要等它最后一步退休才释放：请求表要等这个请求没有在途步了，才放行回收。

## 多 rank

- `sched.workers` 列出每个 rank 的 worker，位置就是 rank。
- 每个 rank 收到同一份计划，只有记录头里的 `rank` 不同：各段只编码一次。
- 所有 rank 都回答了，这一步才退休。token 用 rank 0 的；各 rank 的 `Sampled` 必须完全一样，否则算故障。
- 容量取各 rank 报上来的最小值。所有 rank 都报过以后，才设给账本。

## Resync

- 一步的所有 rank 都回了 `Resync`，这一步就算没跑：
  - token 没算，请求的位置退回去；
  - 条目仍归原来的请求。
- 在它之后已经发出的步也会被拒（worker 收到全量增量之前什么都不跑）。所以下一份计划要等这些步都回来，在没有在途步的时候作为全量增量发出。这时每行的输入 token 都由调度器给出。
- 一个 rank 回 `Resync`、另一个跑了，状态就拼不回去了，算故障：真 TP 下，后者已经推进了自己的 GDN 状态。

## 故障：不冻结

**算故障的情况**：
- worker 回 `StepFailure`；
- 一步超过 `sched.step_timeout_s`（默认 3600 秒）还没有全部回答；
- 各 rank 对同一步的回答不一致；
- 账本在请求运行中被重建了，见下面的"已知缺口"。

**出了故障以后**：
- 不再给 worker 发步；新请求回错误；还在等待准入的请求也回错误；
- 已经占着行的请求保持原样，留给下一次改动引擎的人决定怎么处理（第 5 步）；
- `report()`（`sched.status@1`）给出：哪一步、哪个 rank、原因，以及各有多少请求在跑、在排队；
- 控制通路照常：状态查询、改配置、重载都能做。`quiesce()` 不等出故障的步，所以变更仍然能暂停。

**第 5 步之前的临时办法**：故障标记只放在 `core.sched` 实例里，重载它就会清掉。新实例接手时：
- 请求表里有、自己却不知道的步，就是旧实例丢下的步。这些步里的请求结束并回错误，说明它们在设备上的状态未知；
- 其余请求从最后一次退休的状态接着跑，第一份计划是全量增量。

这相当于"强制继续"这个选项。

## worker 被换（2026-11-09，Codex 评审 #31 后改）

- 一个 rank 的计划或结果走了新连接，多半是 worker 进程被换了，也可能是连接断过、上面的记录丢了。不管哪种，请求在旧 worker 上的状态都没了。
- 这种情况不算故障，因为状态确定是丢了，而不是说不清。处理是：
  - 状态在旧 worker 上的请求当场结束，回错误；
  - 发给旧 worker、它还没回答的步作废；
  - 下一份计划是全量增量，照常服务，不用重载。
- 哪些请求算"在旧 worker 上"：每个请求记着它发给每个 rank 的第一份计划走的是哪一代连接。
  - 计划通道换代时，第一份计划走旧连接的请求都算。
  - 结果通道换代时，在这个 rank 上有过计划的请求都算；同时丢掉计划写端，下一次写入重新查找地址，免得接着往旧 worker 写。
- 同一次替换，两条通道先后都会看到，第二次只做配对，不再处理一遍。所以第一份计划就发给新 worker 的请求不受影响（空闲时换 worker，下一个请求照常服务）。
- 多 rank 时（Codex 第二轮评审）：计划通道先发现的话，发现之前那一步已经发出去了，它是普通增量。
  - 被换的 rank 上的新 worker 要先拿到全量增量，所以会拒绝它；其他 rank 照常跑了它。这种不一致是能解释的，不算故障：这一步算没跑，下一份计划给所有 rank 发全量增量。
  - 已经跑过它的 rank 上，状态往前推进了，所以不能直接回滚重放。按请求分开处理：
    - 退回以后什么都没算过的请求（新请求），在所有 rank 上从位置 0 重新算。从 0 开始时它的状态会先被清掉（`StateStore.clear`），所以没有问题。开重叠时它的头两块都可能这样被拒，结果一样。
    - 进度大于 0 的请求，各 rank 的状态已经不一样了，结束并回错误。这样的请求本来也是在旧 worker 上丢了状态的。
  - 原因不明的 rank 不一致，仍然算故障。
- 第 5 步加上"按 token 历史退回重算"以后，可以在这里把"结束"换成"退回排队"。

## 暂停

- `quiesce()` 等在途步全部回答，最多等 `sched.quiesce_timeout_s`（默认 30 秒）。等不到就抛错：这次暂停失败，变更撤回，引擎照常服务。
- 它还把给前端的 token 写完；一个前端在 `sched.frontend_timeout_s` 内什么都不收，就放弃这一路。

## 前端

- 每个返回地址（`ReturnRoute`）一个发件箱，写的时候不等待。
- 一个前端在 `sched.frontend_timeout_s`（默认 5 秒）内一条都不收，它的请求全部取消，写端丢掉。之后从同一地址来的请求照常服务，不再像 #22 那样留一张 `dead_routes`。
- 坏记录只丢那一条，记一条诊断。

## 配置项

| 键 | 默认 | 绑定 | 说明 |
|---|---|---|---|
| `sched.workers` | `[]` | startup | 每个 rank 的 worker 的 ProcId，rank 0 在前 |
| `sched.frontend_host` | `""` | startup | 往读端 `requests` 写请求的前端在哪台主机 |
| `sched.max_batch` | 32 | startup | 计划最多用多少行，不超过 worker 的 `batch.max_rows` |
| `sched.token_budget` | 2048 | startup | 每步最多算多少 prompt token |
| `sched.overlap` | `false` | live | 一步重叠 |
| `sched.step_timeout_s` | 3600 | startup | GPU 的第一步可能要编译好几分钟 |
| `sched.quiesce_timeout_s` | 30 | startup | 暂停最多等在途步多久 |
| `sched.frontend_timeout_s` | 5 | startup | 前端多久不收 token 就放弃 |
| `sched.idle_wait_s` | 0.01 | startup | 没事做时最多等一条记录多久 |

账本：`alloc.states` 是模型的 `StateDecl` 列表。这一版只支持按页分配的请求条目和每请求一个的槽位，不做前缀快照。

## 给第 4 步 worker 的要求（从测试夹具得出）

- 计划里的每一行都采样，`Sampled` 里一行不少。
- `step()` 返回最后一个回答过的步号。暂停时跟随的 worker 要跑到领头调度器停下的那一步；一直返回 `None` 的话，暂停会超时。真实进程测试就是这样发现的。
- 第 N 份计划还没回答时，可能已经收到第 N+1 份。
- 收到全量增量之前，每份计划都回 `Resync`。持久 batch 重建只丢行和上一次采到的 token，KV 和 GDN 状态在条目里，全量增量之后从原来的位置接着算。
- rank：单元从 `ctx.instance.proc` 拿到自己的 ProcId（前端也是这样用的）。每个 rank 一个进程的角色里，进程号就是 rank，测试夹具就这样取。2026-11-09 更正：原来写的"单元拿不到 ProcId"是错的（Codex 指出）。

## 已知缺口

- **状态单元被重建（第 5 步）**：
  - 变更只要碰到 `sched.reqs` 或 `core.state.alloc` 的代码、清单或配置，它们就会被重新加载，状态全丢：`core.deps` 还不为 `drain_on_reload = true` 的单元生成排空动作（state-changes.md 的实现留在了阶段 2）。
  - 调度器能发现账本在请求运行中被重建，并判为故障；请求表被重建则发现不了。
  - 第 5 步的冲突检查要拒绝这种变更，并报告现状。
- **worker 进程被换**：受影响的请求结束并回错误（见上面"worker 被换"），按 token 历史退回重算还没做。
- **单行上限**（2026-11-09，#30 合入后补上）：调度器用 `[reads]` 读 `worker.batch` 的 `batch.max_context_len` 和 `batch.max_rows`（stage2-contract.md §4.3）。
  - 准入时，prompt 加 `max_tokens` 超过一行容量的请求直接回错误，不再发给 worker 让它报错。
  - 计划最多用 `min(sched.max_batch, batch.max_rows)` 行。
- **配置值的形式**：控制面比较配置的新旧值时，tuple 和 list 不相等。所以客户端传的配置要用 JSON 的形式，否则下一次无关的变更会以为配置变了，把账本重新加载掉。测试里就踩到了这一点（`msgspec.to_builtins` 不会把 tuple 转成 list）。
- **快照对象的开销**：决策点的快照每步给每个请求建一个 `Candidate` 对象，§4.5 担心的正是这种开销。M2a 的规模可以接受，量大了可以改成列。

## 测试

`tests/units/scheduler/`：
- `test_ledger.py`：
  - 联合准入；
  - 延迟释放；
  - 退回没发出去的部分；
  - 开重叠时两步都被拒，位置不会被推到前面。
- `test_scheduling.py`：进程内的单元运行时，两个 rank，夹具 worker 在各自的线程上，全部走 TCP。覆盖：
  - 分块 prefill；
  - FCFS 不插队；
  - prefill 优先；
  - 开重叠和不开 token 一致；
  - 所有 rank 回答才退休；
  - Resync 整批重发；
  - rank 不一致；
  - worker 故障、超时；
  - 暂停失败后照常服务；
  - 取消后延迟释放；
  - 重载调度逻辑后请求接着跑；
  - 故障后重载；
  - 策略决定越界；
  - 账本被重建；
  - 前端失联；
  - 请求太大；
  - 准入守住 worker 每行的容量和行数；
  - worker 被换：空闲时换，以及占着行时换。占着行时换的那一组覆盖了计划通道先发现、结果通道先发现两种顺序，1 个和 2 个 rank，开或不开重叠。
- `test_processes.py`：控制面真的启动调度宿主进程和两个夹具 worker 进程，worker 0 走共享内存，worker 1 走 TCP。开着重叠生成到一半，通过控制接口重载 `core.sched`；同一个请求在新旧两个身份下都收到了 token，和直接算出来的逐个一致。
- 夹具 worker（`units/schedfix/worker`）：
  - 维护持久 batch；
  - KV 页记着里面的 token 和归属，GDN 槽记着推进到了第几个位置；
  - 逐份检查计划；
  - 生成的 token 只取决于 seed、位置和输入 token。

```sh
uv sync --python 3.12 --extra cpu
uv run pytest tests/units/scheduler -q
```
