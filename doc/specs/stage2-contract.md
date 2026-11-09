# 阶段 2 的单元契约：Qwen3.8-27B

- 日期：2026-11-07；2026-11-08 更新共享接口边界与 #21 状态
- 状态：**草案**。GPU 相关的几节（§5.5、§8、§9、§10）等 @gaow0007 审过才算定稿；标“（待定）”的地方还没定。
- 对应：[roadmap](../roadmap.md) 阶段 2；[v0-plan](../v0-plan.md) 5.10–5.13；[modules.html](../modules.html) 的单元清单；GPU 探针报告（PR #17，`doc/evaluations/gpu-probes.md`，含合并前补测的 P9–P11）；[state-changes.md](state-changes.md)；[change.md](change.md) 4.2、4.6、4.7。
- 接口的代码形式在 `engine/core/contracts/`：本文写规则和理由，签名以那里为准。
- 本文定下的事（2026-11-07 和 @wutianyuan1 讨论确定）：
  - 分工（§1）；
  - 契约按“最多提前一步”的重叠调度来设计（§4.4）；
  - 阶段 2 先只做 TP1；
  - GPU 测试放 `tests/gpu/`；
  - 探针报告第 1–10 条建议采纳，第 11 条 v0 不采纳（§13）。

## 1. 范围、分工和里程碑

### 1.1 范围

范围是 roadmap 阶段 2 列的全部内容：
- 层；attention（full attention 和 GDN）；
- 状态种类（paged KV、GDN 的 recurrent state 和 conv state）、权重存储；
- 真实执行后端、`cuda_sm100` 平台、真 CUDA graph；
- 调度逻辑和默认策略、固定长度投机、前缀复用；
- OpenAI 前端。

多 rank 的 TP 配置放在阶段 2 的最后（M2d）。阶段 2 不做 DS 1017 专有的单元（`moe.*` 等）。

### 1.2 分工

| | 负责 | 说明 |
|---|---|---|
| @gaow0007（GPU、CUDA graph） | `platforms.cuda_sm100`、`core.plan`（包括 plan 编译）、`executors.cuda`、attention 和 GDN 的 kernel 包装（`kernels.flashinfer`、`kernels.fla`、对应的 backend）、自己这几块在 GPU 上的验证 | 状态在显存里怎么排（§5.5），由 @gaow0007 决定 |
| @wutianyuan1 和 agent 们 | 其余全部：前端、调度、账本、请求表、每步执行循环、持久 batch、模型、层、kernel 目录和参照实现、状态单元、权重存储、`platforms.torch_eager`、控制面单元的新规则、集成测试和全模型对齐 | Devin 在 SM100 上做集成验证和补测 |
| 本文和 `engine/core/contracts/` | 两边共有 | 团队 PR 由双方 review；同一 RSI agent 可一起修改接口与全部使用者，不增加运行时审批 |

分工的边界划在**可用 CPU 参照实现验证的接口**上：
- 平台有 `torch_eager` 和 `fake`；
- kernel 有 PyTorch 参照实现；
- graph 有 eager 路径。

所以 @wutianyuan1 这边不用等 GPU 部分，就能把整条链路跑通；@gaow0007 按契约和契约测试实现，不需要了解引擎全局。这几块的契约测试由 @wutianyuan1 这边先写好。临时固定输出等替身只放测试夹具，真实实现可用后删除；参照 kernel 和真实 CPU eager 执行则保留用于数值对齐。

### 1.3 里程碑

| 里程碑 | 内容 | 主要靠谁 |
|---|---|---|
| M2a | 小尺寸、随机权重的 Qwen3.8 结构（例如 4 层：3 层 GDN + 1 层 full attention），参照 kernel，eager，在 `torch_eager` 上把前端 → 调度 → worker → 前端整条链路跑通，和同尺寸的 HF 参照逐 token 对齐 | @wutianyuan1 |
| M2b | SM100 上的 TP1，换成真 kernel，仍然 eager，全模型和 HF 参照在容差内对齐 | 两边集成 |
| M2c | decode graph：graph 和 eager 按下面的口径一致；能分片的变更在准备阶段按分桶切片重录，不能分片的按 §9.4 处理 | @gaow0007 为主 |
| M2d | 前缀复用、MTP 投机、流式输出；五组对比；一步重叠；一种多 rank 配置 | @wutianyuan1 为主 |

验收沿用 roadmap，另加一组对比（§4.4）：
- 全模型和 HF 参照在容差内对齐；
- 快速测试层里，“批量对单条”“前缀命中对不命中”“开投机对不开”“decode graph 对 eager”“开重叠对不开”五组对比按下面的口径一致；
- 能服务流式请求。

**对比的口径**（2026-11-09 @wutianyuan1 定，暂定，以后可能再修）：数值和逻辑算对就行。kernel 本身带来的差异不纳入验收：bf16 的舍入，以及 batch 怎么拼带来的累加顺序不同（§6 `batch_invariant`）。测试这样写：
- fp32 下 logits 在容差内；
- greedy 采到的 token，只在第一名和第二名的 logit 之差大于容差时才要求相同；差得更少时选了不同的 token，不算失败；
- bf16 只报告从哪里开始不同，不判失败。

### 1.4 分支和合并（2026-11-08，提议，待两人确认）

以下是团队协作流程，不是 RSI 运行时的人工审批规则；个人分支的实际命名和启用另行决定。共享接口能否原地修改以 §3 为准。

- 每人一条个人分支，`dev/wutianyuan1` 和 `dev/gaow0007`，都从 master 拉出。自己的 PR 合进自己的分支，自己审。CI 对所有分支的 push 和 PR 都跑；合进个人分支的 PR 不发PR 通知（`notify.yml` 只管 master）。
- 两边共用的东西不走个人分支，直接 PR 到 master，两边各有一个人审过才合：
  - `engine/core/contracts/` 和本文；
  - 框架本体（`engine/framework.toml` 列出的模块，如 PR #18）；
  - `pyproject.toml`、`uv.lock`（torch 的两个 extra，§11.2）、`.github/`。

  这些东西如果在两条分支上各改各的，汇合时冲突最难解，而且另一边的代码是照着旧版本写的。`AGENTS.md` 的目录地图两边都会改，冲突好解，留在个人分支上。
- master 上一有这类改动，两条个人分支都尽快把 master 合进来。用 merge，不用 rebase：个人分支已经推上去，别人在上面开 PR，rebase 会改写它的历史。
- **汇合点**：每个里程碑结束时；以及一边要用另一边的东西时。步骤：
  1. 先合的分支开 PR 到 master，CI 通过、对方看过，用 merge commit 合进去。不用 squash：squash 以后个人分支的历史和 master 对不上，下次汇合还会冲突。
  2. 后合的分支先把新的 master 合进来，在自己分支上解决冲突、跑通 CI，再开 PR 到 master。
  3. 两条都合完，各自从 master 接着做（个人分支 fast-forward 到 master）。
- **先后顺序**：谁的改动是对方的前提，谁先合。通常是 @wutianyuan1 先合：GPU 这边的单元插在整条链路里，它们的 GPU 测试要用到链路；链路反过来只要 CPU 替身就能跑。
- **第一个依赖**：M2a 的 worker 通过 `plan@1` 跑 eager，而 `core.plan` 归 @gaow0007。所以 `core.plan` 的 eager 路径（只有分桶 0，不录 graph）要先做，在 M2a 之前单独汇合一次。在那之前，@wutianyuan1 这边在测试里用一个只走 eager 的替身。

## 2. 进程和单元

Qwen3.8、TP1 时的进程和单元如下。目录按单元 id 放在 `engine/` 下（例如 `core.sched` 放在 `engine/core/sched/`），每个目录一份 `unit.toml`。“@gaow0007”一栏打勾的归 @gaow0007，其余归 @wutianyuan1。

| 进程（角色） | 单元 | 种类 | 提供 | 需要 | @gaow0007 |
|---|---|---|---|---|---|
| 前端 `frontend` | `frontend.http` | 逻辑 | — | `frontend.tokenizer@1`、`frontend.detok@1` | |
| | `frontend.tokenizer` | 逻辑 | `frontend.tokenizer@1` | | |
| | `frontend.detok` | 逻辑 | `frontend.detok@1` | | |
| 调度宿主 `sched`（`leads_epochs`） | `core.sched` | 逻辑 | `proc.loop@1`（带 `quiesce`） | `sched.reqs@1`、`ledger@1`、各决策点 | |
| | `policies.*` | 逻辑 | `policy.<决策点>@1` | | |
| | `sched.reqs` | 状态 | `sched.reqs@1` | | |
| | `core.state.alloc` | 状态 | `ledger@1` | 各类池先作为内部实现，按实际独立替换需求再拆分 | |
| worker（每 rank 一个） | `core.worker` | 逻辑 | `proc.loop@1`（带 `quiesce`） | `worker.batch@1`、`plan@1`、`model@1` | |
| | `worker.batch` | 状态（缓存） | `worker.batch@1` | | |
| | `core.plan` | 逻辑，录 graph | `plan@1` | `model@1`、各 backend、各状态单元 | ✓ |
| | `executors.cuda` | 逻辑 | `executor@1` | | ✓ |
| | `executors.eager` | 逻辑 | `executor@1`（和 `executors.cuda` 二选一） | `worker.batch@1`、`model@1`、各 backend、各状态单元 | |
| | `models.qwen38` | 逻辑 | `model@1` | `layers.*`、`state.weights@1` | |
| | `layers.*`（linear、embedding、lm_head、norm、rope、attn 外壳、gdn 外壳、sampler） | 逻辑 | `layers.<名字>@1` | `kernels.catalog@1` | |
| | `core.kernels` | 逻辑 | `kernels.catalog@1` | | |
| | `kernels.reference` | 逻辑 | 往目录里登记参照实现 | | |
| | `kernels.flashinfer`、`kernels.fla` | 逻辑 | 往目录里登记 solution | | ✓ |
| | `attn.backend.flashinfer`、`gdn.backend.fla` | 逻辑 | `attn.backend.*@1`、`gdn.backend.*@1` | 状态单元、`worker.batch@1` | ✓ |
| | `core.state.holder` | 状态 | `state.holder@1`（显存预算、状态池容量） | | |
| | `state.kv_paged`、`state.gdn_slots` | 状态 | `state.kv.paged@1`、`state.gdn.slots@1` | `state.holder@1` | 布局 ✓ |
| | `core.state.weights` | 状态 | `state.weights@1` | `state.holder@1` | |
| | `core.dist`、`dist.groups` | 逻辑、状态 | M2d 才要 | | |
| 控制面 | 已有的四个单元 | | | | |

- 控制面单元要加的规则：
  - `core.deps`：重录 graph 的推导（graph 的依赖登记成派生产物，依赖变了就重录：能分片的在准备阶段，不能的在暂停期内，§9.4），以及容量变更的影响；dry-run 报出重录造成的停顿；
  - `core.manifest`：单元不许自己建 stream（§10.4）。
- 新平台 `platforms.torch_eager` 和 `platforms.cuda_sm100` 都属于框架本体，要加进 `engine/framework.toml`。

## 3. 最小共享交接面（2026-11-08 更新）

共享定义只描述实际跨单元、跨进程交接的数据和必要调用。内部对象布局、算法、队列、缓存索引和辅助函数归实现所有。不能因为一个功能需要类型提示，就把整套业务对象搬进 contracts。

`core.contracts` 是位于所有单元目录之外的库：共享定义若放进提供方的单元目录，其他单元 import 它就会触发 `manifest.cross_unit_import`。

- 新增算法、换 kernel、改调度策略，沿用现有交接方式时直接改实现。
- 交接内容或语义确实改变时，才更新共享定义；单个 RSI agent 可以同时修改接口及全部使用者。
- 所有使用者在同一次变更中更新时，可以直接修改定义。需要不同版本并存时再新增分段或版本；不要求每个功能都新增文件、分段或 `@2`。
- 原地修改还要处理旧格式的在途记录：新接收方必须能正确解码并按原语义处理队列里的旧消息；否则先停止该流的生产并排空，或用新分段/版本区分。调度器与 worker 的计划、结果由 §4.4 的 `quiesce()` 收齐；前端与调度器的流在步循环之外，其暂停入队、排空或兼容解码要在 M2a 单独落实。现有 msgspec 映射结构忽略未知字段、给缺失字段补默认值，只解决编码层面的兼容，仍需检查字段含义。
- 两位负责人审核是当前团队的 PR 协作流程，不是 RSI 运行时的人类审批条件。`apply` 仍按依赖、暂停和变更协议执行。
- `__init__.py` 不重新导出各模块。模块按真实交接边界隔离，避免无关重载；`REACHES` 测试记录实际依赖。
- 修改前后用依赖分析确认影响范围。没有运行实例时可检查 ImportGraph；有实例时用 dry-run。它是工程检查，不是审批。

当前保留：

| 模块 | 实际交接 |
|---|---|
| `requests`、`sampling`、`columns` | 前端、调度器和 worker 使用的消息与采样数据 |
| `serving` | 前端的 token 返回地址（`ReturnRoute`），随每条请求记录一起发给调度器 |
| `records` | 调度器发给 worker 的步计划和结果 |
| `state`、`storage`、`tensor` | 模型向存储/账本提供的当前状态声明，以及 buffer 交接 |
| `kernels`、`model` | kernel 注册和选择，以及模型向 plan/权重存储提供的数据 |
| `plan` | worker 获取 plan，以及 backend 与持久 batch 的必要调用 |
| `ledger`、`policy` | 调度器调用账本的最小调用集；调度器的两个决策点（准入和排序、这一步怎么组成）的快照和决定（2026-11-08，第 3 步） |

本轮收缩：删除共享 `ledger.py`（账本插件尚是本组内部实现），`common_prefix` 移到 `core.state.alloc.prefix`；删除 graph 的内部 `GraphDep`、强制公开的重录流程、重复的可选状态动作 Protocol，以及未实际交接的状态/模型预留字段。请求表与账本的内部字段不建立共享镜像。状态声明先保留 request/slot，跨层借用、host 放置和分区等在实际实现需要交接时再加入。

## 4. 每步记录（两边最重要的接缝）

代码：`core/contracts/records.py`（调度 ↔ worker）、`requests.py`（前端 ↔ 调度）、`sampling.py`、`columns.py`。

### 4.1 为什么要先定这个

vLLM 从 V0 重写到 V1，CPU 开销是主要原因之一。2024 年 9 月 vLLM 公布的 profile 里（Llama 3 8B、一张 H100），GPU 执行只占 38%，HTTP 占 33%，调度和准备输入占 29%。V1 的做法有两条：
- 相邻两步的状态大部分相同，就复用，只发变化的部分；
- 能挪出核心循环的事，都挪出去。

SGLang v0.4 把调度和执行重叠起来：GPU 在跑第 N 步时，CPU 已经在安排第 N+1 步。vLLM 的 Model Runner V2 设计文档也承认，V1 的异步调度是后来补上的，所以留下了复杂度。

我们比它们更需要这一条：调度宿主和 worker 是两个进程。不重叠的话，每一步的关键路径上都要多一次跨进程往返，再加上调度器的全部 CPU 时间。所以契约从一开始就按“最多提前一步”来设计，哪怕阶段 2 默认跑的是同步循环。

[step-overhead.md](../evaluations/step-overhead.md) 的结论（“保持同步”）只针对一件事：进程宿主的循环用 asyncio 还是同步。它测的是空 `step()` 外面的框架开销，不涉及 CPU 和 GPU 的重叠。同步的循环完全可以做重叠：mini-sglang 的 [overlap_loop()](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/scheduler/scheduler.py) 就是一个普通的 `while` 循环，靠异步 launch 和晚一步收结果。

### 4.2 记录和分段

每步记录是开放记录（[ipc.md](ipc.md) §3）。epoch 就是步号，由调度方分配。

| 方向 | schema | 分段 | 内容 |
|---|---|---|---|
| 调度 → worker | `sched.step_plan@1`（每 rank 一份） | `sched.rows@1`（`RowDelta`） | 持久 batch 的增量：新增的行（请求、采样参数）、删掉的行；`full` 表示整体重发 |
| | | `sched.batch@1`（`BatchPlan`） | 这一步每行算几个 token、从哪个位置开始、输入 token；`from_previous`；分桶 |
| | | `state.ops@1`（`StateOps`） | 每种状态新分给各行的条目（页、槽位），以及条目拷贝（前缀命中时恢复快照、块边界打快照） |
| worker → 调度 | `worker.step_result@1` | `worker.sampled@1`（`Sampled`） | 每行采到的 token（开投机时是接受的长度加一） |
| | | `worker.stats@1`（`StepStats`） | 这一步的 CPU 时间、GPU 时间，以及开始前 GPU 空了多久 |
| | | `state.capacity@1`（`Capacity`） | 各状态种类可用的条目数，启动时和已支持的容量变更生效后发 |
| | | `worker.resync@1`（`Resync`） | 持久 batch 被重建了，要一份全量 |
| 前端 → 调度 | `frontend.requests@1` | `frontend.batch@1`（`Requests`） | 新请求（已分词）、中止 |
| 调度 → 前端 | `sched.tokens@1` | `sched.events@1`（`TokenEvents`） | 这一步各请求的新 token、结束原因，以及这一步跑在哪个身份下 |

四条都走每步记录流：有序、不丢、单生产者、背压。

`BatchPlan` 的 token 怎么按行拆（2026-11-08，PR #19 评审）：
- `from_previous = 1` 的行只算 1 个 token，不从 `tokens` 里取；
- 其余的行按行序，从 `tokens` 里依次各取 `num_tokens` 个；
- 所以 `tokens` 的长度正好是这些行的 `num_tokens` 之和。

worker 在测试和调试模式下按这几条检查每份计划。检查的代码归 worker 实现，不放进契约（§3：契约只放交接的数据和必要调用）。

投机的草稿 token 不在这一版里。草稿放在哪、验证步的 `num_tokens` 怎么数，到 M2d 做投机时再定，是否新增分段或版本按 §3 判断（§4.4 第 7 条）。

### 4.3 持久 batch

- worker 里有一个状态单元 `worker.batch`，保存所有在跑请求的副本。每行在 host 上有：请求、各种状态的条目 id、采样参数；在显存里有：位置、各状态的条目表（block table）、上一次采到的 token。
- 调度每步只发增量（`RowDelta`、`StateOps`），不重建整个 batch。
- 它是缓存：重载以后是空的，于是发 `Resync`，调度就发一份 `full` 增量，所以重载它不需要排空（`drain_on_reload = false`）。
- **显存里的表只在 step stream 上按顺序更新**（入队拷贝），不从 host 直接同步写。这样上一步如果还在 GPU 上跑，它读到的仍然是 launch 时的值。
- **重建以后**（2026-11-08，PR #19 评审）：
  - 重建丢掉的不只是行和条目 id，还有每行上一次采到的 token。重建发生在暂停期内，调度器 `quiesce()` 时已经收完了所有结果，所以这些 token 它都有。
  - 一行被重新加入以后（`full` 增量会重新加入所有行），它下一次跑的输入 token 由调度器给出，跑过一次以后才能再用 `from_previous = 1`。恢复以后隔了几步才再排进 batch 的行也是这样。`full` 增量里的 `StateOps` 给每行分配它持有的全部条目，不只是新的。
  - 收到 `full` 之前，worker 不跑任何计划，每份都回 `Resync`、不带 `Sampled`。调度器把这些步当作没跑过：token 没算，拷贝没做，请求的长度退回到最后一个回了 `Sampled` 的步。这些步分出去的条目仍归原来的请求，`full` 里会再分一次。开了重叠时，`full` 之前已经发出的那份计划也会回 `Resync`，这是预期的。
  - worker 碰到 `from_previous = 1` 却没有可接的 token，也回 `Resync`，不跑。
  - 2026-11-09 更新（#30 评审）：新建的持久 batch 也算“重建以后”，收到第一份 `full` 之前什么都不跑。调度器启动后的第一份增量本来就是 `full`。
  - 采样不在 worker 里留随机状态：客户端没给 seed 的话，调度器在接纳请求时补一个，worker 按（seed, 位置）取随机数。所以重建前后采到的 token 一样，和 batch 怎么拼也无关。
  - 开投机时，下一步要验证的草稿如果留在显存里，也一起丢掉；重建后每行的第一步不带草稿。
  - 契约测试（M2a，`torch_eager`）：请求 decode 若干步以后重建 batch，再接着生成，和不重建的对照逐 token 一致；要覆盖重建后隔了几步才再排进 batch 的行，以及开重叠的情况。
- **上限**（2026-11-09，#30 评审）：持久 batch 最多 `batch.max_rows` 行，每行最多 `batch.max_context_len` 个 token（按整页留表）。超出上限的增量，`apply` 在改动任何东西之前就报错，不回 `Resync`：再发一份 `full` 也装不下。调度器通过 `[reads]` 读这两个键，接纳时拒绝装不下的请求（#31）。

### 4.4 一步重叠

1. **深度**：调度器收到第 N−1 步的结果之后，才能发第 N+1 步。所以没收到结果的计划最多两份：一份在跑，一份在排队。
2. **future token**：`from_previous = 1` 的行，输入 token 就是 worker 上一次给这一行采到的 token，在显存里直接接上，不回 host。不开投机时，位置是确定的（加一），调度器自己算得出来。
3. **提前分配**：第 N+1 步要的页，调度器先按“请求还会继续”分出去。如果请求在第 N 步已经结束，这些页按第 5 条延迟释放。
4. **结束晚一步**：请求在第 N 步采到了停止 token，调度器要等第 N 步的结果才知道，这时第 N+1 步已经发出去了。多算出来的那个 token 由调度器丢掉，不发给前端（`TokenEvents` 的说明）。前端在反分词时发现停止字符串，发 `AbortRequest`，处理方式相同。
5. **延迟释放**：账本的 `release(request, last_step)` 只做标记。等最后一个用到这些条目的步的结果到了（`retire(step)`），它们才回到空闲表。否则新请求拿到这个槽位，会被还在跑的旧步写坏。
6. **暂停**（PR #18）：
   - 调度器的 `quiesce()`：不再发新计划，把已发计划的结果都收完，返回最后一个做完的步号。
   - worker 的 `quiesce()`：等 GPU 上的工作都做完。
   - 领头的调度器先停，跟随的 worker 跑到同一步再停。所以执行阶段卸掉的东西，不会还有在途的工作在用。
7. **投机解码**：开投机时，下一步的位置取决于这一步接受了几个草稿 token，这个数只在 GPU 上才知道。v0 规定开投机时不重叠。以后位置也可以写成引用，具体记录形式在 M2d 决定，是否新增分段或版本按 §3 判断。
8. **同步循环是特例**：收到第 N 步的结果才发第 N+1 步，记录完全相同。阶段 2 默认走同步循环；M2d 加开关 `sched.overlap`（live 键），并加一组对比：开重叠和不开重叠，按 §1.3 的口径一致。2026-11-08 更新：开关和在途步队列在第 3 步就做了（默认关），对比先用测试夹具跑（`engine/core/sched/README.md`）。
9. **可观测**：`StepStats` 给出每步的 CPU 时间、GPU 时间、GPU 空闲时间。RSI agent 靠它判断瓶颈在不在 CPU，再决定开不开重叠、怎么调。

### 4.5 按列编码

- 每行一个值的字段都编码成列（`Column`）：一段带类型的字节串，用标准库的 `array.array` 构造，读的时候可以零拷贝（`torch.frombuffer`）。一批 256 个请求时，不要每步给每个请求建一个 Python 对象，那是 vLLM V0 走过的老路。
- 每步只有新增的请求（`AddedRow`、`NewRequest`）是逐个的对象，它们的数量很小。
- 小端字节序。

## 5. 状态

代码：`core/contracts/state.py`（声明）、`storage.py`（存储单元）；`Binding` 在 `tensor.py`。账本实现与辅助算法在 `core/state/alloc/`，不放进共享 contracts。

### 5.1 声明

每层声明自己要存的东西（`StateDecl`），存储单元和账本内部的状态池用同一份声明：
- 种类；
- 每个条目里有哪些张量（shape、dtype）；
- 当前寿命：`request`（按 token 分页）或 `slot`（每请求固定条目）；
- 每个条目对应几个 token；
- 每隔多少 token 打一次快照；
- 每个 rank 实际交接的形状由该 rank 的模型/存储实现协商。

### 5.2 存储单元（worker 里，`state.<kind>@1`）

- 必须实现：
  - `declare`；
  - `bind(layer) -> Binding`：交出这一层的 buffer，并带一个绑定代数；
  - `capacity`；
  - `scratch`；
  - `copy`：条目拷贝，在 step stream 上做；
  - `invalidate`。
- 可选：`grow`、`refit`、`relocate`、`export_state` / `import_state`，用 `capabilities.actions` 和 `[state] migrate` 声明。`grow` 仅保留为实现可声明的动作，不要求阶段 2 实现在线扩容，也不承诺地址不变或免重录。
- **绑定代数**：交出去的 buffer 只要有一个换了地址、shape 或 dtype，代数就加一；同地址换内容不加。graph 在 replay 前，每个提供者只比对一个整数（§9.3）。
- **scratch 条目**：capture 前的 warm-up 会真的跑起来，GDN 的 recurrent state 会被推进，KV 会被写入。如果用在途请求的条目来 warm-up，它们的状态就坏了（探针 P6）。所以每种状态都留出 scratch 条目，永远不分给请求。
- **条目编号**（2026-11-09，#30 评审）：账本分出去的编号是 `0 .. capacity - 1`，scratch 条目的编号在这个范围之外（`scratch()` 给出）。条目在 buffer 里放在哪是存储单元自己的事。SGLang 布局的两个存储单元把条目 `i` 放在页或槽 `i + 1`，和行 `r` 在表里是 `r + 1` 一样；页或槽 0 是 SGLang 的 padding，也就是 scratch 条目，编号 `-1`。持久 batch 的表里存的是页号和槽号，所以 padding 行的 0 指向的正是 padding。

### 5.3 联合账本与内部状态池（调度宿主里，`ledger@1`）

2026-11-08 更新：以下是 M2d 前缀缓存的设计目标。池与联合准入算法归 `core.state.alloc` 实现所有，不预先要求拆成独立单元；共享 `AccountingPlugin` 已删除。需要独立替换插件时，再围绕实际交接提取必要接口。本 PR 只保留已有的 `common_prefix` 辅助算法，账本实现由后续 PR 提供。2026-11-08 更新：第 3 步接入了 #22 的不带前缀缓存的账本，调度器调用的最小调用集在 `core/contracts/ledger.py`。

`core.sched` 调用状态单元 `core.state.alloc` 的 `ledger@1`，仍然是真实跨单元交接。M2a 接入时，将调度器实际需要的最小调用集定义到 `core.contracts`；本节的准入、预留/快照、发布、取消、延迟释放和退休语义继续约束它，随当期功能实现相应调用。每类池的 `reusable`、`declare`、`set_capacity` 等内部细节留在账本实现里。

- 只管 id，不碰显存。
- `reusable`（给定长度以内最长能从哪里恢复，按各自的对齐规则）、`admit`（带前缀接纳）、`need` / `take`（再要条目）、`snapshot`（规划时预留快照）、`publish`（结果回来后登记新块）、`cancel`（退回没发出去的那部分计划拿到的东西）、`release` / `retire`（延迟释放，§4.4 第 5 条）、`check`（不变量）。
- `core.state.alloc` 做两件事：
  - 联合准入：一个请求要的每种状态都分得到，才接纳它；
  - 联合前缀匹配：找每种状态都能从那里恢复的最长前缀（`common_prefix`）。
- **联合前缀不是各自最长命中的最小值**（2026-11-08，PR #19 评审）。例：KV 每页 16 token，最长命中 96 token；GDN 每 64 token 一个快照，现存 64 和 128。最小值是 96，可是 GDN 在 96 没有状态，只能从 64 恢复。
- 算法：
  - `reusable(block_hashes, limit)` 返回这种状态在 `limit` 以内能恢复的最长长度。KV：最长连续命中以内的任一页边界；GDN：现存快照里不超过 `limit` 的最大一个。
  - 从 `limit` 开始，取各种状态答案的最小值，再拿这个值去问，直到没有一种状态再压低它。
  - 停下的长度每种状态都支持；而且每种状态的答案都不会低于真正的共同最长长度，所以停下的就是它。
- 单位是 token：
  - `block_hashes[i]` 覆盖前 (i+1)·B 个 token。B 是 `core.state.alloc` 的哈希块大小，默认等于 KV 页大小。
  - 每种状态的对齐（页大小、快照间隔）都必须是 B 的倍数，`declare` 时检查。
  - `limit` 是 prompt 长度减一，再向下对齐到 B：至少留一个 token 要算，才有 logits。
- 算法测试在 `tests/units/scheduler/test_prefix.py`，覆盖不同的快照间隔、不同的最长命中、快照被淘汰的情况。`common_prefix` 逐个检查每种状态的答案：只看最小值的话，一个合法的答案会盖住另一个超出范围的答案。

**前缀缓存的生命周期**（2026-11-08，PR #19 评审）：
- 前缀索引（块哈希 → 条目）由账本内部的各类状态池持有，因为每种状态对齐和存前缀的方式都不同；联合账本负责编排。
- 条目分三种：
  - 空闲；
  - 被请求持有：可以同时被几个请求持有；预留的快照槽由它快照的那个请求持有，直到发布；
  - 已缓存：发布过，没有请求持有，可以淘汰。

  状态池只在要分出条目时才淘汰，按最久未用。被持有的条目、待拷贝的源条目都不淘汰。
- **接纳**：先用 `common_prefix` 定下长度 L，再对每种状态调 `admit(request, block_hashes, L, new_tokens, step)`。
  - 它返回两样东西：按表顺序排好的条目（先是复用的，再是新分的），放进 `StateAssign`；这一步之前要做的拷贝，放进 `StateCopy(before)`。
  - 要么全给，要么返回 `None`、什么都不拿。只要有一种状态返回 `None`，这个请求就不接纳，已经接纳了它的那几种状态调 `cancel(request, step)`，拿到的东西当场退回。
  - 不能用 `release(request, step)` 撤销（2026-11-08，PR #19 第三轮评审）：`release` 要等这一步的结果，可这一步如果谁都没接纳，可能根本不会发出去，预留的槽和快照引用就一直放不掉。`cancel` 只用于还没发出去的那部分计划；规划时拿了条目、最后又没排进计划的行，也用它退回。
  - KV：复用的页原地读，请求一直持有到 `release`。
  - GDN：新分一个槽，拷贝源是缓存的快照。快照被持有到这一步的结果回来，所以拷贝跑完之前不会被淘汰。
- **快照和发布分两步**（2026-11-08，PR #19 第三轮评审）。开了重叠时，第 N 步的结果还没回来，调度器就要规划第 N+1 步，而 N+1 的输入 token 还在 worker 上（`from_previous`），所以规划时算不出覆盖它的块哈希。只有拷贝必须在规划时定，哈希可以晚一步：
  - **快照**（规划时）：每行这一步的 chunk 末尾在 `upto`，调 `snapshot(request, upto, step)`。不需要哈希。
    - GDN：`upto` 正好是快照间隔的倍数时才打快照（§5.4）。状态池预留一个空闲或可淘汰的槽当目标，返回 `StateCopy(after)`；预留不到就不打。
    - KV：页本来就留着，不用拷贝。
  - **发布**（结果回来以后）：先 `retire(step)`，再对这一步跨过块边界的请求调 `publish(request, block_hashes, upto)`。这时 token 都已知，哈希算得出来，新块当场就能被别的请求命中。
    - 不能更早发布：别的请求可能读到这一步还没写、甚至永远不会写的条目。
    - KV：登记请求自己的页。
    - GDN：登记它在 `upto` 以内预留的快照，快照变成已缓存，不再由请求持有。
  - 重叠一直不用中断，同步循环走的也是同一套调用。
- **结束**：调度器先发布最后一个结果算完的块，再 `release(request, last_step)`。`release` 等最后一步的结果回来才生效；生效以后，发布过、又没有别的请求持有的条目变成已缓存，其余的回到空闲表，包括没来得及发布的快照槽。
- **`Resync`**：
  - 回了 `Resync` 的步，调 `retire(step, ran=False)`。这一步什么都没写，它预留的快照槽放回空闲表。
  - 接纳时的拷贝排在这种步里的请求，退回排队：调度器先 `release` 它，再按它的 token 历史重新接纳。
- **账本验收测试**（M2d，`core.state.alloc` 的 KV/GDN 联合记账）：
  - 请求 A 算完一段前缀、发布、结束；请求 B 命中这段前缀，拿到的是 A 的页和快照槽；B 的拷贝和使用没做完之前，这些条目不会被回收或淘汰。
  - 联合准入里第一种状态接纳了、第二种拒绝，这一步不发：第一种的容量和缓存引用当场恢复，随后一个小一点的请求仍能接纳。
  - 开重叠，推迟前一步的 `Sampled`，让下一步正好落在快照边界：token 未知时不登记哈希；最后发布的位置、哈希和状态，和同步循环逐项一致。

### 5.4 Qwen3.8 的状态

- `kv.paged`：16 层 full attention，每层的 K、V。24 个 q 头、4 个 kv 头，head dim 256。
- `gdn.slots`：48 层 GDN，每层的 conv state `[槽, 10240, 3]` bf16 和 recurrent state `[槽, 48, 128, 128]` fp32。TP1 时每个请求约 147 MiB，256 个并发就是约 37 GiB（探针报告第 9 条）。槽位数在启动时由 `core.state.holder` 按显存预算确定，不要求在线扩槽位。
- 前缀复用：GDN 状态只在块边界有快照。快照占同一个池里的槽位，打快照、恢复快照都是条目拷贝（`StateCopy`）。所以 `gdn.slots` 的 `reusable` 只能对齐到快照边界。
- **快照只能在一步的末尾打**（2026-11-08，PR #19 评审）：
  - GDN 的状态在一步之内原地往前推，`StateCopy(after)` 只拷得到这一行在这一步 chunk 末尾的状态。
  - 例：一个 256 token 的 chunk 跨过了 64、128、192。步末时这几处的状态已经没有了，不能把步末的状态当成它们存进缓存。
  - 规则：chunk 的末尾正好落在快照边界上才打快照，chunk 里面跨过的边界不打。
  - 想在哪里留快照，就由分块决策点把 chunk 切在那里。默认 chunk 长度取快照间隔的倍数，并把 prompt 里最后一个快照边界切成一个 chunk 的末尾。这样，共用同一段 system prompt 的后来请求能从最靠后的边界恢复。
  - decode 每步只算一个 token，正好落在边界上的那一步打快照。开投机时，一步接受几个 token 事先不知道，所以 v0 开投机时 decode 期间不打快照。
- MTP：config 里有 1 层 MTP，它的 KV 是 `kv.paged` 的另一层声明。

### 5.5 状态布局：严格照 SGLang（2026-11-09）

**2026-11-09 @wutianyuan1 和 @gaow0007 定：KV cache 和 GDN 状态池的组织方式不变，严格照 SGLang**（`MHATokenToKVPool`、`MambaPool`，§5.5.8）；配置项、默认值和启动时的兼容性检查也照 SGLang（§5.5.3）。阶段 2 每种状态只有这一种物理布局，不做开放的布局注册，也不要求第二种物理布局（HND 等）。§5.5.1 的分层、§5.5.4 和 §5.5.5 的规则继续适用；§5.5.7 是 SGLang 的配置空间记录，以后加布局时照它来。

以下是 @gaow0007 的原设计（2026-11-09），按上面的决定改过 §5.5.2、§5.5.3、§5.5.6、§5.5.8。存储单元由 @wutianyuan1 这边实现，GPU kernel 的约束由 @gaow0007 提供。

#### 5.5.1 分开描述语义、布局和分配

| 层次 | 内容 | 责任 |
|---|---|---|
| 状态语义 | K/V、MLA latent、GDN recurrent/conv；逻辑轴、尺寸和数值精度要求 | 模型与状态声明 |
| 物理布局 | 轴顺序、shape、stride、dtype、alignment、辅助 buffer，以及跨层组织 | 存储实现与 kernel 侧协商 |
| 分配策略 | page size、槽位容量、窗口回收、快照与 scratch 预留 | 存储持有方与账本 |
| 消费者能力 | 可读写的布局、dtype、stride、页大小和执行模式 | backend、copy、传输等实际消费者 |

- full attention、SWA、GDN 可以使用不同的池和布局。SWA 规定访问窗口与回收语义，不直接规定页内 NHD/HND；MLA 也不套用普通多头 KV 的布局枚举。
- 以每层 K/V 分开存储为例：NHD 是 `[pages, tokens_per_page, kv_heads, head_dim]`，HND 是 `[pages, kv_heads, tokens_per_page, head_dim]`。K/V 是否合并、不同层是否打包，是另外的物理组织选择。
- GDN recurrent 的候选布局用有语义的轴名表达，例如 K-last `[slots, value_heads, value_dim, key_dim]`；conv 单独选择 `[slots, channels, history]`（DS）或 `[slots, history, channels]`（SD）。不能因 K/V 维度恰好相等就省略轴语义。
- §5.4 的 shape、FP32 recurrent 和显存数字是当前模型的基线。改变精度、conv 历史容量或投机临时状态后，需要重新计算预算并验证数值；持久状态精度和 kernel 的累加精度分别记录。
- page size 是逻辑缓存页的 token 数，分配字节数与对齐由实际存储实现确定。

#### 5.5.2 布局：照 SGLang，不做开放注册（2026-11-09 改）

- 阶段 2 每种状态的物理布局就是 SGLang 的那一种（§5.5.8），写在存储单元里。不建布局注册表，`StateDecl` 和 `Binding` 里也不加布局名：只有一种布局时，这些字段没有消费者（§3）。
- 以后要加第二种布局，照 SGLang 的方式加：做成存储单元的一个配置项（例如 SGLang 用 `SGLANG_USE_HND_KVCACHE` 选 HND），消费它的 kernel 在启动校验时检查（§5.5.3）。到那时再按 §3 补需要交接的字段。
- 原设计里仍然成立的两条：形状对得上不等于物理布局兼容；`permute()` 得到的视图可能不连续，只有消费者接受对应的 stride 时才能直接用。

#### 5.5.3 配置项与启动时的检查：照 SGLang（2026-11-09 改）

配置项只对应 SGLang 已有的参数，默认值和 `auto` 的含义也照 SGLang。都是 startup 键：

| agxinfer 的键 | SGLang | 默认值 |
|---|---|---|
| `state.kv.page_size` | `--page-size` | 16 |
| `state.kv.dtype` | `--kv-cache-dtype` | `auto`：等于模型的 dtype |
| `state.gdn.ssm_dtype` | `--mamba-ssm-dtype` | `auto`：模型 config 里的 `mamba_ssm_dtype`，没有就是 f32（`mamba2_state_dtype`） |
| `state.gdn.conv_dtype` | 环境变量 `SGLANG_MAMBA_CONV_DTYPE` | bf16（SGLang 的 conv state 不跟模型 dtype） |
| `attn.backend`；`attn.prefill_backend`、`attn.decode_backend` | `--attention-backend`；`--prefill-attention-backend`、`--decode-attention-backend` | `auto`；按阶段的两个键为空时跟 `attn.backend` |
| `gdn.backend`；`gdn.prefill_backend`、`gdn.decode_backend` | `--linear-attn-backend`；`--linear-attn-prefill-backend`、`--linear-attn-decode-backend` | `auto`；同上 |

- backend 的名字就是 kernel 目录里 solution 所属的那一组（§6），例如 `reference`，以后的 `flashinfer`、`triton`。
- **显式指定的 backend 是硬约束**：这一组的 solution 必须覆盖该阶段要用的每个 definition，并且适配这个模型的 shape 和平台，否则启动失败；不退回参照实现。
- **`auto`**：按 kernel 目录的规则选（§6），选不到就用参照实现，并在日志里标出 fallback。
- dtype 的约束照 SGLang 在启动时检查（如 `handle_linear_attn_backend`：SM100 上 flashinfer 的 GDN decode / verify 要求 `mamba_ssm_dtype` 为 bf16），不兼容就报错，不悄悄降精度。这类检查随对应的 kernel 一起接入。
- 解析在 worker 启动时做一次，结果打一条 INFO 日志：平台；每种状态的 dtype 和页大小；prefill、decode 两个阶段各自的 backend 和每个 definition 选中的 solution，是否 fallback；verify 标为 disabled。运行中不再换。
- fp32 的对齐测试把三种状态的 dtype 都显式配成 f32；默认配置下 conv state 是 bf16，和 SGLang 一样。

#### 5.5.4 前缀复用、投机与 buffer 交接

- 完整 KV 页的 prefix hit 通过页 ID 和引用计数共享，与页内 NHD/HND 解耦。共享页不得直接续写；部分页复用需要 copy-on-write 或明确限制为完整页。页大小变化同时影响前缀对齐和索引。
- GDN 缓存快照恢复到请求自己的可写状态。快照、运行状态、投机临时状态和 capture scratch 可以采用相同物理格式，但所有权、可写性和回收时机分别管理。不能把关闭投机期间的前缀快照当成已经解决拒绝草稿后的状态恢复。
- binding 交接必须能核对实际 buffer 的轴语义、shape、stride、dtype、alignment 和有效访问范围。量化辅助 buffer 也属于绑定的一部分。
- §5.2 的 generation 规则还须覆盖 stride、布局解释及辅助 buffer 绑定的改变，即使主 buffer 地址不变也要使不兼容的旧 graph 失效。同地址仅更新内容可保持绑定代数，但前缀数值有效性仍需单独判断。
- 元数据更新、状态写入、copy 和 replay 必须有明确的 stream 顺序；固定地址不保证在途读取安全。跨 stream 时建立依赖，并保留异步操作所需的源、目标和输出，直到最后一次使用完成。

#### 5.5.5 配置变更与在线迁移

布局相关配置通过变更计划生效，不作为任意时刻可改的 live 开关。计划要报告迁移范围、临时显存、停顿、前缀缓存处理和 graph 重录；必要能力未实现时，执行前拒绝。

| 变更 | 所需处理 |
|---|---|
| 更换 kernel，兼容已有 binding | 保留存储；检查 solution/代码依赖，必要时重录 graph |
| 物理 NHD ↔ HND、conv DS ↔ SD、跨层重新打包 | 显式迁移并验证，或排空后重建；更新绑定并处理旧 graph |
| 仅改变无复制视图 | 验证消费者的 stride 支持；重新检查绑定与 graph，不默认零影响 |
| page size 改变 | 协调存储、页表、账本、前缀对齐与索引；不能仅修改 tensor shape |
| 状态 dtype/量化格式改变 | 验证数值并转换或重建；明确前缀快照是否失效，不自动继续复用 |
| 已支持的状态池容量变更 | 按实际资源变化检查绑定；重新分配或改变视图结构时递增 binding generation、使受影响 graph 失效，并按 §9.4 处理状态迁移和重录；无实现则拒绝 |

- 只有数值语义保持且引用/索引能正确重映射时，迁移才可保留前缀缓存；否则明确失效或排空。
- 旧 buffer、graph 与复制源必须等在途使用结束后再释放。迁移或 capture 失败时清理候选资源；只有确认旧状态完整才能恢复服务，否则进入恢复流程，不能报告切换成功。

#### 5.5.6 分阶段实现与验收

- 首版只实现 SGLang 的这一种布局（§5.5.8），不要求第二种物理布局，也不要求 PD、量化格式或其他 attention 类型（2026-11-09 改）。
- M2a 用 CPU 参照验证页/槽寻址、copy、clear、padding 条目，以及 §5.5.3 里不兼容的显式配置在启动时被拒绝；M2b 在 SM100 验证真实 kernel，覆盖 prefill → decode（2026-11-09 改）。
- M2c 覆盖地址相同但 stride 改变时的 graph 失效、capture scratch 隔离、候选失败清理，以及在途工作完成后释放旧资源。
- M2d 随功能落地覆盖 prefix hit/miss、共享页续写保护、GDN 快照恢复、投机部分接受后的 recurrent/conv 一致性，以及启用的 overlap/传输路径。
- 比较逻辑结果，不比较不同布局的原始字节顺序。使用非对称内容和非平凡 stride；参照实现支持时用不等的 K/V 维度，避免方阵掩盖转置错误。保持 dtype 和计算路径时验证布局转换不改变语义；精度变化另设明确容差与长序列验证。
- 延续 A/B/C 方法：A 缺少待验收能力或存在原缺陷，B 为正确实现；C 植入忽略 stride、遗漏 generation 更新、共享页被写入或提前释放等缺陷。加强测试直到 A/C 失败而 B 通过。性能测试计入状态写入、适配复制和同步成本，不只测 attention kernel。

#### 5.5.7 SGLang 配置空间与兼容约束清单（2026-11-09）

以下是 [SGLang](https://github.com/sgl-project/sglang) 的源码静态分析记录。本节的远程源码入口固定到 [`ff762d6`](https://github.com/sgl-project/sglang/tree/ff762d62f8d236e95903620ff0b7e82ac89890db)，便于定位与后续复核。原始分析所用源码副本未记录上游提交，部分文件与该链接版本不同，因此不声称下表与链接版本逐项一致；实际接入时须按选定版本复核能力约束。下表区分可配置选项、原始检查到的代码约束和 agxinfer 的设计决定；没有运行 GPU 性能或全组合正确性实验。SGLang 可通过注册扩展选项，枚举的是原始检查中的内置项。

**attention/backend 维度。** prefill 与 decode 可分别选择；draft 模型和 linear-attention verify 还有自己的 backend。名字出现在 choices 中不代表适用于每种模型、dtype 或设备。

| 家族 | 检查到的内置 backend/结构 | 对布局的含义 |
|---|---|---|
| 通用与常规 attention | `triton`、`torch_native`、`flex_attention`、`fa3`、`fa4`、`flashinfer`、`trtllm_mha`、`hpc_ops` | MHA/GQA/MQA 保存 K/V；页内布局、stride 和量化格式由具体路径约束；`hpc_ops` 的选项注释限定 Hopper、page size 64 |
| MLA | `flashmla`、`trtllm_mla`、`cutedsl_mla`（旧别名 `tokenspeed_mla`） | 压缩 latent/位置分量及可能的量化元数据；不能套用普通多头 NHD/HND 选择 |
| 稀疏、压缩和模型专用路径 | `dsa`（旧别名 `nsa`）、`qsa`、`dsv4`（旧别名 `compressed`）、`ernie55`、`unlimited_ocr_v2`、`llada2_cfg_flashinfer`、`minicpm_flashattn`、`minicpm_flashinfer` | 可能增加 indexer、压缩比或模型专用状态；逐项提供布局实现与消费者能力 |
| 其他设备路径 | `aiter`、`wave`、`intel_amx`、`ascend`、`intel_xpu` | 仅作为扩展空间；不纳入 SM100 首版能力承诺 |
| GDN/KDA 等 linear attention | `triton`、`cutedsl`、`flashinfer`、`flashkda`、`nvidia_kda`、`ptx_kda`、`helion`、`intel_xpu`；verify 另有 `nv_cutedsl` | 多项是 KDA 专用或阶段受限，不能全部用于 GDN；recurrent/conv 的轴、dtype、槽位 stride 分别匹配 |
| SWA / full+SWA / full+GDN | 属于模型状态组合，不是 NHD/HND 的别名 | full、window、recurrent/conv 分池或通过兼容视图组合；窗口回收与页内顺序分开决定 |

远程参考：SGLang [arg_groups/choices.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/choices.py) 的 `ATTENTION_BACKEND_CHOICES`、`DRAFT_ATTENTION_BACKEND_CHOICES`、`LINEAR_ATTN_KERNEL_BACKEND_CHOICES`；[mem_cache/kv_cache_configurator.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/mem_cache/kv_cache_configurator.py) 的 pool 分派。

**prefix cache 与其他配置轴。** 这些维度与 attention 组合形成候选空间，但不是任意笛卡尔积都合法。

| 配置轴 | 检查到的选项或模式 | 需要纳入布局校验的内容 |
|---|---|---|
| GPU prefix cache | `disable_radix_cache`；默认 unified radix、pure SWA 的专用路径；session 引用；可注册 cache backend | 命中/关闭不自动更换物理布局；共享、窗口保留和引用计数由缓存实现负责 |
| GDN prefix state | `auto` / `no_buffer` / `extra_buffer` / `extra_buffer_lazy`；路径快照数量限制；可选 int8 checkpoint 池 | 活跃状态和缓存快照可能使用不同 dtype/池；快照恢复需要匹配位置、格式及转换精度 |
| 分层缓存 HiCache | host `cache` / `buffer_only`；write-back / write-through / selective；IO 为 `direct` / `kernel` / `kernel_ascend` | GPU 布局、host 布局、IO 路径分别匹配；buffer-only 要有 storage backend |
| host cache 布局 | `layer_first`、`page_first`、`page_first_direct`、`page_first_kv_split`、`page_head`、`page_unified` | 这是 host 跨层/页的组织方式，不是 GPU 页内 NHD/HND；需要显式传输描述 |
| 外部 prefix 存储 | LMCache、FlexKV、external linker、自定义注册 backend | 入口可扩展；恢复格式、版本和数值身份必须相容，不能只比较 token 哈希 |
| GPU 池组织 | 默认逐层池、`enable_page_major_kv_layout`、`enable_unified_memory` | 跨层打包/统一字节池可产生非连续层视图；不等同于 HND，需要 stride-aware 消费者 |
| KV 精度 | `auto`、BF16（`bf16`/`bfloat16`）、`fp8_e5m2`、`fp8_e4m3`、`mxfp8`、`nvfp4`、`fp4_mx_block16` | FP8/FP4 recipe、scale buffer、打包与对齐必须一起描述，不能只替换 dtype |
| recurrent 精度 | `float32`、`bfloat16`、`float16`；某些路径有随机舍入 | 持久状态、快照、kernel 累加精度分开验证；不是所有 backend 都支持这些选项 |
| speculative decoding | 关闭或 EAGLE/EAGLE3/NEXTN/STANDALONE/NGRAM/DFLASH/DSPARK/UNO，以及注册算法；另有 draft backend/dtype、草稿长度和 top-k | target/draft 分别匹配；verify 的状态写入、拒绝恢复、scratch 容量及 tree/chain 能力必须明确 |
| page size / 并行 | 页大小、TP/CP/DCP、可注册的 `cp_kv_layout` | 本 rank 的 head/latent 分片、页表语义和 cache/传输能力必须匹配 |
| PD / CPU offload | prefill/decode 分离、host 迁移、decode 回退 | 以具体 pool 和传输实现的格式能力为准，不从 attention 名称推导支持 |
| chunk / graph / overlap | chunked prefill、eager/graph、一步 overlap | 不自行决定页内布局；会约束 scratch、地址稳定性、元数据时序和快照边界 |

远程参考：SGLang 配置字段 [memory.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/memory.py)、[model.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/model.py)、[exec_.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/exec_.py)、[spec.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/spec.py)、[parallel.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/parallel.py)、[disagg.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/disagg.py)、[schedule.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/fields/schedule.py)；[mem_cache/registry.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/mem_cache/registry.py) 中的 `default_radix_cache_factory`。

**已检查到的具体约束。** 这些是原始检查中具体实现路径的约束，不是 HND、SWA 或某个算法的普遍限制。

| 实现路径 | 代码证据 | 对 agxinfer 的处理 |
|---|---|---|
| 普通 FlashInfer wrapper | [layers/attention/flashinfer_backend.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/attention/flashinfer_backend.py) 多处以 `NHD` 初始化；部分直连路径要求 direct NHD | NHD 作为该路线基线；HND 不能仅靠改配置宣称支持 |
| TRT-LLM MHA | [trtllm_mha_backend.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/attention/trtllm_mha_backend.py) 的 `_reshape_paged_kv_cache` 经 `paged_kv_view`/`permute` 交出 HND 形状，可能仍是 NHD 物理 stride | 区分 kernel 形状和物理排列；适配能力按具体调用验证，不默认整个池必须复制，也不默认任意 stride 都可用 |
| MHA pool 的 PD / CPU offload | [memory_pool.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/mem_cache/memory_pool.py) 的 `get_contiguous_buf_infos`、`get_cpu_copy`、`load_cpu_copy` 路径拒绝 `use_hnd` | 未实现并验收 HND 传输适配前，HND 与这些路径组合 fail-fast |
| MXFP8 pool | [memory_pool.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/mem_cache/memory_pool.py) 中的 `MHATokenToKVPoolMXFP8` 拒绝 `SGLANG_USE_HND_KVCACHE` | 量化是独立布局能力，不能继承普通 HND 的支持结论 |
| page-major / unified pool | [arg_groups/kv_cache_hook.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/kv_cache_hook.py) 按完整、线性、prefill/decode backend 校验；部分 prefill 需要 gather/scatter 连续状态 | 默认保持逐层池；跨层组织作为独立扩展，公开适配成本 |
| CP sharded padded | [mem_cache/registry.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/mem_cache/registry.py) 拒绝 radix cache 或 streaming session；[page_interleave_pool.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/mem_cache/page_interleave_pool.py) 的相应 KV sharding 路径只接受 NHD | 明确标为不支持的组合，不把 prefix cache 开关当成万能兼容项 |
| HiCache IO | [arg_groups/hicache_hook.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/hicache_hook.py) 将 `page_first + direct` 解析为 `page_first_direct` | GPU/host 布局分别解析；agxinfer 对显式配置冲突报错，只有 auto 可以选择兼容替代 |
| SM100 FlashInfer GDN | [arg_groups/attention_hook.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/arg_groups/attention_hook.py) 的该 decode/verify 路径要求 BF16 recurrent state；[gdn_triton.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py) 声明 K-last 池 | 保留 FP32 参照路线；请求该 FlashInfer 路径时显式选择并验收 BF16，不静默降精度 |

#### 5.5.8 阶段 2 的布局：严格照 SGLang（2026-11-09 改）

原设计的默认值表和 auto 的选择顺序（逐层独立池、`[P,16,Hkv,D]`、HND 扩展等）按 §5.5 开头的决定作废，改成下表。这就是 #30 实现的布局（2026-11-08），代码不变。

| 状态 | 布局（SGLang） | dtype（§5.5.3） | 说明 |
|---|---|---|---|
| full-attention KV（`kv.paged`） | `MHATokenToKVPool`：每层一份 K、一份 V，`[容量 + page_size, kv 头数, head_dim]`，按 token 槽寻址（物理上是 NHD）；槽 0 所在的页做 padding | `state.kv.dtype`，默认等于模型的 dtype（bf16） | page size 16；前缀缓存只发布完整页，没写满的页归请求自己（§5.5.4） |
| GDN recurrent（`gdn.slots` 的 temporal） | `MambaPool`：`[GDN 层, 槽 + 1, v 头, v_dim, k_dim]`（K-last）。所有 GDN 层叠放在一块里，每层的切片是连续的；槽 0 做 padding | `state.gdn.ssm_dtype`，默认 f32 | temporal 是 `[v_dim, k_dim]`，和 HF 的 `[k_dim, v_dim]` 互为转置；参照 kernel 按 SGLang 的排法写，和 HF 比时转置（`tests/units/kernels/test_reference.py`） |
| GDN conv | `MambaPool`：`[GDN 层, 槽 + 1, conv_dim, kernel - 1]`（DS） | `state.gdn.conv_dtype`，默认 bf16 | |
| 持久 batch 的表 | `HybridReqToTokenPool`：`req_to_token` `[行 + 1, 最大上下文]` int32，以及行到 GDN 槽的映射 | | |
| HND、跨层打包、量化 KV、MLA、SWA、HiCache、PD、offload、CP | 阶段 2 不做 | | 以后照 SGLang 对应的开关加（§5.5.2、§5.5.7） |

**组合验收**：§5.5.3 每一种不兼容的显式配置都有一条启动失败的测试；prefix 开/关、eager/graph、投机和重叠的交叉，随对应功能在 M2c、M2d 覆盖。

## 6. kernel 目录

代码：`core/contracts/kernels.py`。schema 和 flashinfer-bench 的 Definition / Solution 对齐。

- **definition** 写算什么：名字带版本、语义说明、PyTorch 参照实现（必须有）、容差、结果和 batch 怎么拼是否无关。参照实现在 `torch_eager` 上跑，是 CI 的基准，也是检验 solution 的标准。
- **solution** 写怎么算：用哪个库的哪个版本、支持哪些 shape、能不能录进 graph、需要哪些平台能力、要不要进程级初始化、能不能跨机。
- 选择：编 plan 时按 definition 加 shape 选定，eager 路径每次调用都查。选不到合适的就退回参照实现，并在 plan 和结果里标出来。选中的 solution 是 plan 的按值依赖。
- kernel 单元通过 `ctx.register` 登记到目录里，层只按 definition 调用，不直接调具体的 kernel。
- **实现（2026-11-08）**：
  - `core.kernels`（`engine/core/kernels/`）提供 `kernels.catalog@1`。kernel 单元登记 `Definition` 和 `Implementation`（solution 加函数本身）；重载时登记随撤销表撤掉，新实例再登记。
  - 选择：符合 shape 和平台能力的非参照 solution 里按 id 取第一个；没有就用参照实现，`Selected.fallback` 为真。`Definition.reference` 是参照 solution 的 id。
  - definition 的名字和 solution 的 id 在目录里唯一，重复登记会报错，所以查找结果和登记顺序无关。重载时旧实例的登记先撤掉，新实例再登记，不会撞名。参照 solution 的 id 带版本（`reference.gemm.linear@1`），同一 definition 的新旧版本可以同时登记（2026-11-09）。
  - `kernels.reference`（`engine/kernels/reference/`）登记 Qwen3.8 要的全部 definition 和 PyTorch 参照：`norm.gemma_rms`、`norm.rms_gated`、`pos.rope`、`act.silu_and_mul`、`gemm.linear`、`embed.lookup`、`kv.store`、`attn.extend`、`attn.decode`、`gdn.conv1d_fn`、`gdn.conv1d_update`、`gdn.gating`、`gdn.chunk_delta_rule`、`gdn.recurrent_update`、`sample.seeded`（都是 `@1`）。
  - **2026-11-09 改为从 SGLang（`e838360`）搬运**：每个 definition 的签名就是它对应的 SGLang kernel 的签名，参照实现是 SGLang 的 native 路径，或 SGLang 用来检验 CPU kernel 的 torch 参照（`test/registered/cpu/test_mamba.py`、`test_causal_conv1d.py`，最初来自 vLLM 和 fla）。attention 是 `TorchNativeAttnBackend` 的 SDPA，采样是 `Sampler` 加 `multinomial_with_seed`，murmur hash 从 Triton 改写成 torch。原来自己写的 `pos.rope_neox`、`attn.paged`、`gdn.conv1d`、`gdn.delta_rule` 已删除。
  - 和 HF `qwen3_5` 对照测试；conv 和 delta rule 按 SGLang kernel 的接口在状态池上按槽读写，可以接着上一块的状态继续算。采样按（seed, 位置）取随机数。
  - **和 batch 怎么拼是否无关**（`Definition.batch_invariant`，默认否，2026-11-09 更正）：
    - 声明为是的只有按构造成立的：`attn.extend` / `attn.decode`、`gdn.conv1d_fn`、`gdn.chunk_delta_rule` 逐个序列算，lookup 和 store 只搬数据。decode 的 conv / delta rule 一步和采样是整批一个算子，不声明。
    - norm、RoPE、激活、linear、gating 是一个 torch 算子作用在整个 batch 上，PyTorch 会随行数换计算路径和累加顺序，不声明。实测 CPU fp32 的 linear，同一行单独算和放进 batch 里算，最后几位不同；两个 logit 很接近时，greedy 会选出不同的 token。
    - 所以五组对比不要求逐位一致，按 §1.3 的口径（2026-11-09）。
  - 测试：`tests/units/kernels/`，对照 HF 的实现、分块等于不分块；声明和 batch 怎么拼无关的参照，单独算和放进 batch 里算逐位相同。

## 7. 层、模型、权重

代码：`core/contracts/model.py`。

- 逻辑张量 = 格式（插件）+ 放置（数据）。层不持有权重，只从权重存储拿视图，所以 refit 是原地换内容，graph 不用重录。
- 模型单元在启动时给出 `Assembly`：用哪些层、每层存什么状态、有哪些权重。各项尺寸从 checkpoint 的 config 读，不写死。
- Qwen3.8：64 层，48 层 GDN、16 层 full attention，排列是 3 层 GDN 加 1 层 full attention 重复。M2a 用同样的结构、小尺寸、随机权重。
- **实现（2026-11-09，worker 循环以下的 eager 路径，全部从 SGLang `e838360` 搬运后适配）**：

  | 单元 | 提供 | 来自 SGLang |
  |---|---|---|
  | `core.state.weights` | `state.weights@1` | `DefaultModelLoader` / `DummyModelLoader`：`safetensors_weights_iterator`、`initialize_dummy_weights`；参数放在这个状态单元里，模型重载不丢；`refit` 原地换内容，不留副本，abort 时重新读原来的来源（和 `update_weights_from_disk` 的回滚一样） |
  | `models.qwen38` | `model@1` | `Qwen3_5ForCausalLM`（TP1，无投机、量化、LoRA、融合 kernel）：参数名用 SGLang 的（融合后的 `qkv_proj`、`gate_up_proj`、`in_proj_qkvz`、`in_proj_ba`），`load_weights` 原样搬运，作为 loader 登记到权重存储 |
  | `layers.linear`、`embedding`、`norm`、`rope`、`activation`、`sampler` | `layers.<名字>@1` | `LinearBase`、`VocabParallelEmbedding` 和 `LogitsProcessor` 的 LM head、`GemmaRMSNorm`、`RMSNormGated`、`RotaryEmbedding`、`SiluAndMul`、`Sampler`；都按 definition 调目录 |
  | `layers.attn`、`layers.gdn` | `layers.attn@1`、`layers.gdn@1`（`Backend`） | `RadixAttention` + `TorchNativeAttnBackend`；`RadixLinearAttention` + `GDNAttnBackend` 的普通路径。`prepare` 就是 `init_forward_metadata`；MIXED 步走 extend |
  | `state.kv_paged`、`state.gdn_slots` | `state.kv.paged@1`、`state.gdn.slots@1` | `MHATokenToKVPool`（每层 `[size + page_size, kv 头, head_dim]`，槽 0 所在页是 padding）；`MambaPool`（conv `[层, 槽 + 1, conv_dim, K-1]`、temporal `[层, 槽 + 1, HV, V, K]` fp32，槽 0 是 padding），`copy` 即 `copy_from`，`clear` 即 `clear_slots`；条目 `i` 是页或槽 `i + 1`（§5.2） |
  | `worker.batch` | `worker.batch@1` | `HybridReqToTokenPool`：`req_to_token`、行到 GDN 槽的映射；记录里的行 `r` 是表里的 `r + 1`，条目 `i` 是页或槽 `i + 1` |
  | `executors.eager` | `executor@1` | `ModelRunner`：从 position 0 开始的行先清 GDN 槽（`mamba_clear_indices`）、恢复快照、组装 `input_ids` / `positions`、各 backend 准备、前向、在每行最后一个 token 的位置采样 |

  - 接缝：worker 循环只调 `executor@1`（`core/contracts/execution.py`），持久 batch、backend、模型、采样器都在它后面。`executors.eager` 和 `executors.cuda` 是同一个服务的两种实现，一个 worker 角色只加载其中一个（和 `policy.compose@1` 可以有几种实现一样），不算撞名（2026-11-09，#30 评审）。
  - 契约随之补了：`WeightStore.register` / `load`、`Model`、`PersistentBatch.check` / `sampling` / `record`、`StateStore.clear`。
  - 每一行都采样（SGLang 也是）。prefill 中间块采到的 token 由调度器在调 `RequestTable.sample()` 之前丢掉。
  - 测试（`tests/units/executors/test_eager.py`，CPU fp32）：贪心生成和 HF 逐 token 相同，logits 在 1e-4 内；批量对单条、分块 prefill 对不分块、batch 重建（先回 `Resync`，收到全量增量后接着生成）对不重建、快照恢复对从头算，token 相同，logits 在容差内；条目从 0 开始分（和账本一样），每步之间改写 scratch 条目，token 不变。

## 8. attention 和 GDN 的 backend（@gaow0007 审）

代码：`core/contracts/plan.py` 的 `Backend`。以下补充记录 2026-11-09 的评审结论；是待实现的契约，不表示当前接口和 GPU 路径已经全部落实。模型/backend 的计算边界、分阶段执行及状态处理参考 [SGLang](https://github.com/sgl-project/sglang)（源码入口及版本范围见 §5.5.7），资源管理和配置变更接入 agxinfer 的 state store、worker 与 plan。

**职责总则：backend 准备元数据、组织计算及状态更新；worker 保证执行顺序和配置切换边界；plan 管理 graph 的捕获、选择和失效。** backend 不在每步调用中自行切换配置或重录 graph。

### 8.1 layout 与各阶段 backend/kernel 的联合选择（2026-11-09 讨论确认）

2026-11-09 补充：阶段 2 每种状态只有 SGLang 的那一种布局（§5.5），下面说的显式配置就是 §5.5.3 那张表里的键，解析和日志按 §5.5.3 做。

- **显式 layout 是硬约束。** 用户指定 layout、其余选项为 `auto` 时，参考 SGLang 在对应设备、模型和精度下的默认搭配，从 agxinfer 已注册且已验收的组合中选择。遵守 §5.5.8 的布局和精度约束；不能为了套用默认 backend 而更换显式 layout、降低精度或关闭功能。SGLang 的默认搭配是选择依据，不代表 agxinfer 已支持该组合。
- **启动或配置变更时统一解析、校验完整组合。** 一次解析启用阶段的 prefill/decode/verify backend 与具体 kernel/solution，并校验模型语义、设备、物理 layout/stride、dtype、page size，以及启用的 prefix、speculation 和 graph 能力。未启用的 verify 标为 disabled；以后启用时重新校验。不能让各阶段独立选择后，等执行时才发现状态不兼容。
- **跨阶段状态兼容性属于组合校验。** 不同 kernel 可以共享满足其约定的状态；需要适配时，适配路径必须已实现并验收，且明确转换发生的位置。GDN 同时检查 conv/recurrent 状态、快照恢复和投机部分接受所需的中间状态。形状相同不足以证明语义或物理布局兼容。
- **不兼容的显式配置直接 fail-fast。** 没有合法 auto 组合也失败。在候选池分配、capture 和新配置生效前完成能力校验，报出冲突阶段、用户请求及所需能力。配置变更被拒绝时保留当前有效组合；不静默切换显式 backend/layout。§6 的 kernel 参照回退不能绕过这里的显式约束。
- **auto 必须输出明确的解析日志。** 启动以及每次相关配置变更解析成功时，输出 INFO 级结构化记录，至少包括下表；不能只记录 `backend=auto`。候选结果与已生效结果明确区分，生效时关联配置版本；校验失败则输出冲突诊断。日志按组合记录，避免每层、每 token 重复打印。

| 日志内容 | 要求 |
|---|---|
| 请求与解析结果 | 保留用户原始配置，逐项标明 explicit / auto / default 及最终有效值 |
| 各执行阶段 | prefill/decode/verify 各自的 backend、kernel/solution 标识及版本；未启用阶段明确标为 disabled。阶段内按 shape 分派时列出已选择的规则，编 plan 后可追溯到实际 solution |
| 状态格式 | 实际物理 layout、dtype、page size、相关 shape/stride 和 kernel 视图；GDN 分别列出 conv/recurrent 格式 |
| 适配与选择原因 | 是否需要转换、采用哪条适配路径，以及 auto 为什么选择该组合；使用的默认策略版本可追溯 |
| 配置上下文 | 设备、模型/状态类别、相关功能开关、配置版本，以及 candidate / active 状态 |

### 8.2 模型层与 backend 的计算边界

| 操作 | Full attention | GDN |
|---|---|---|
| 输入/输出投影 | 模型层 | 模型层；输入投影生成卷积前 QKV、递推 gate 输入 `a/b` 和输出 gate 输入 `z` |
| Q/K normalization | 当前模型所需的 Q/K RMSNorm 在模型层侧，与 RoPE 一起在进入 backend 前完成 | conv + activation 之后做逐 head L2 normalization，由 backend 负责；不能与 RMSNorm 混同 |
| 状态计算与更新 | backend 组织当前 K/V 写入及 attention 计算 | backend 统一组织 conv、递推 gate 变换、gated delta-rule，以及 conv/recurrent 两种状态更新 |
| 输出 normalization/gating | 若模型需要，由模型层侧完成 | backend 返回后，在模型层逐 value head 执行 RMSNorm，再乘输出 gate 的激活值，最后做输出投影 |

- GDN 的 `a/b` 在 backend 内转换为递推所需的衰减和更新系数；`z` 用于模型层的输出 gating，不直接更新 recurrent state。归一化范围、epsilon、scale、gate 激活和运算顺序属于模型语义，不能由 backend 任意改变。
- **职责边界不要求每项对应一个 kernel。** Q/K L2 normalization 可以融合进 recurrent kernel，也可以在 backend 内先执行，再关闭后续 kernel 的重复归一化；conv/recurrent 可拆分实现或融合。所有路径保证 normalization、scale、gate 变换各执行一次。
- 模型声明权重，权重存储持有并加载；层和 backend 使用绑定视图。conv 权重、`A_log`、`dt_bias` 在 backend 内参与计算，不意味着由 backend 分配或加载权重。
- **GDN 是整体的执行、更新和验证边界。** 改 conv 的实现也属于 GDN backend 变更，无需同时修改 recurrent 代码，但须检查两阶段衔接、各执行阶段和状态恢复。即使 shape 不变，conv 语义改变也可能使已有 recurrent state 失效；必须声明兼容、迁移、重新计算或排空方案，不能只重录 graph 后继续使用旧状态。
- Full attention 的 KV 位置分配、引用共享和写前复制由调度/存储层安排。backend 消费已分配的写入位置，调用存储写入能力或使用融合 kernel；不得自行分配 page 或覆盖仍共享的 prefix 内容。

远程参考：SGLang [models/qwen3_next.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/models/qwen3_next.py) 中的 `Qwen3GatedDeltaNet`、[layers/radix_linear_attention.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/radix_linear_attention.py)、[gdn_backend.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/attention/linear/gdn_backend.py) 与 [gdn_triton.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/attention/linear/kernels/gdn_triton.py) / [gdn_flashinfer.py](https://github.com/sgl-project/sglang/blob/ff762d62f8d236e95903620ff0b7e82ac89890db/python/sglang/srt/layers/attention/linear/kernels/gdn_flashinfer.py)；mini-sglang [layers/attention.py](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/layers/attention.py) 和 [attention/fi.py](https://github.com/sgl-project/mini-sglang/blob/9a91cfafe754aa85daee49998176275667eb58f2/python/minisgl/attention/fi.py) 展示 full attention 的层/backend/存储分工。mini-sglang 链接固定到 `9a91cfa`；该版本没有 GDN 实现，不能作为 GDN 状态恢复方案的依据。

### 8.3 各阶段执行与状态操作顺序

- prefill、chunked prefill、decode、speculative verify 可以选不同 kernel，参考 SGLang 的分阶段分派；选择与兼容性由 §8.1 一次校验。阶段切换不得重新解释已有状态；需要转换时使用已验收的显式适配路径。
- 调度层决定前缀恢复位置、快照边界和 speculative 接受长度；state store 负责分配、复制和生命周期；backend 负责计算更新，以及生成选定算法所需的中间状态。worker 组织这些操作的先后关系。
- 普通步的逻辑顺序为：校验配置/plan/绑定及输入，完成槽位或页的分配交接，恢复所需状态并准备本步元数据，执行 forward，最后生成需要发布的快照。恢复与元数据准备的具体提交顺序可由实现安排，但两者都必须在依赖它们的 forward 之前完成；快照不得读到尚未完成的更新。
- Prefix 恢复必须让所有层对应同一个逻辑 token 边界；每个 GDN 层的 conv/recurrent 状态必须成对一致，恢复到请求独占槽后再更新。按当前 `StateCopy` 契约，`after` 只能复制本步 chunk 结束时的状态；不能把跨过的内部边界冒充为已保存快照。
- Verify 的临时结果与已接受状态分开处理。仅接受部分 draft token 时，必须获得接受边界的 conv/recurrent 状态，并保证 full attention 的有效长度/页映射不暴露拒绝后缀；不能直接保留整个 verify 的最终状态。中间状态保存或恢复重算的具体方案见 §8.7，普通 chunk-end copy 不能替代该能力。

### 8.4 每步 prepare / forward 与 stream

- **`prepare` 在 graph 外执行，不限于 CPU 操作。** 它从持久 batch 和 `BatchPlan` 准备页表、长度、槽号等元数据，可调用库的 planning 接口、提交 H2D copy 或 GPU 元数据更新。页表可在显存增量维护，只更新变化的行。它不追加本步 K/V，不推进 GDN conv/recurrent 状态，也不接管 worker 安排的 prefix restore。
- **`forward` 同时支持 eager 与 capture。** 它执行层计算和状态更新；固定形状、稳定地址及 capture-safe 是 graph 路径的要求。每步 `prepare` 与配置变更中的候选资源准备、warm-up/capture 是两类工作，后者由变更流程与 plan 管理。
- 首版正常步使用 worker 指定、平台提供的 step stream。batch/backend 元数据写入、状态复制和 forward 按其依赖顺序提交；backend 和 state store 不自行创建 stream。需要辅助 stream 的扩展须明确 event 依赖，worker 的 quiesce 必须覆盖相关工作。
- 一步 overlap 可以让 CPU 准备 N+1，而 GPU 执行 N。共享固定地址 buffer 的设备提交顺序必须是 `metadata[N] → step[N] → metadata[N+1] → step[N+1]`，不能在其他 stream 上提前覆盖。异步 H2D 的 host 源 buffer 在复制完成前也不能复用。
- 库的 `plan/replan` 若修改在途工作使用的 workspace，必须保证 stream 顺序或隔离资源；无法证明安全时，在安全点等待完成后执行。不能把“在 graph 外调用”当成可并发改写资源的保证。
- 日常执行不要求每步 CPU/GPU 同步。候选 graph 的 warm-up/capture 则按 §9.4 暂停新提交，等待在途及相关辅助 stream 工作完成，使用 scratch 和平台提供的 capture stream；capture 完成后再恢复 serving。这不是后台 capture 与同进程 serving 并行的承诺。

### 8.5 Graph 依赖与 replay 前检查

- backend 固定元数据 buffer 通过 `Binding` 交出；状态、权重和持久 batch 的绑定仍由各自提供者交出。buffer 地址、shape、stride、dtype、layout 等变化按 §5.5、§9.3 更新依赖；bucket、选中的 solution、backend 代码和捕获时使用的配置属于相应的按值依赖。
- 页号、槽号、长度等内容变化，仅在具体路径已验证为运行时读取、且未超出捕获容量或改变执行结构时，允许原地址更新并复用 graph。不能仅以 batch size 相同判断 graph 可复用。
- P5/P6 证明的是被测 FlashInfer 路径在其捕获条件内重新 plan 后，可处理同 batch 大小的不同序列长度；不能推广成所有 backend 的 plan 结果都不构成按值依赖。workspace 地址、kernel 选择或执行结构改变时，仍须重新检查，必要时重录。
- worker/plan 在提交本步状态更新前完成输入和执行方案校验，replay 前保留依赖检查；backend 不得在检查后偷偷改变绑定。失效 graph 不得启动；是否选择其他 bucket 或已验收的同配置 eager 路径，由 worker/plan 决定。

### 8.6 Serving 期间的配置变更与失败处理

- **一个 step 使用同一套有效配置、backend、状态绑定和 plan。** worker 在提交前确定它们；不得在 `prepare(A, N)` 与 `forward/replay(A, N)` 之间切到 B，也不能在同一步的不同层间混用版本。
- 候选配置 B 的出现不使仍满足 A 依赖的 graph A 立即失效。准备 B 不得覆盖 A 的配置、元数据、workspace 或活跃请求状态；warm-up/capture 使用隔离 scratch，新旧 graph 使用 §9.2 规定的独立显存池。兼容的请求状态不要求全量复制，但不能拿来试跑候选计算。
- 支持独立候选资源且满足 §9.4 分片前提时，可在安全窗口中逐 bucket 准备 B，窗口之间继续服务 A。只能原地修改资源、或需要重载现有 backend 等单元时，把相关动作推迟到最终暂停阶段；不假设当前框架支持同一单元新旧实例并行服务。
- 最终切换由变更协议组织：停止跨边界的新提交，quiesce 并收集在途结果，完成必要的状态迁移，再使 B 的配置、绑定和执行方案一起生效。多 rank 遵守共同边界。具体重录策略、停顿预算和显存峰值留在 §9。
- B 的状态兼容且 eager 路径已验收时，可以只准备常用 bucket，切换后其余 bucket 走 **B 的 eager**，再在安全窗口补录；不能回退到依赖 A 的旧 graph。未使用 eager 的配置必须等所需 graph 就绪后才能切换。
- 候选准备失败且未修改 active 资源时，清理候选并保留 A。切换或执行中失败则按变更/恢复流程处理；conv 已更新而 recurrent 失败等情况属于部分状态更新，不能直接重试或报告已恢复。确认回滚/重算完成后才能继续，否则使受影响状态失效并进入恢复流程。
- 旧 graph 及其依赖只有在无在途使用、且变更协议不再需要它们回滚时才能释放。记录候选校验、准备、就绪、生效或失败，以及暂停耗时、就绪 bucket 和 eager 覆盖范围；配置“已接受”不等于“已生效”。

### 8.7 待落实接口与验收边界

以下细节尚未定稿；实现前补齐实际交接，不为内部算法增加整对象镜像或通用大接口。

| 待落实项 | 当前边界与建议 |
|---|---|
| Mixed batch | `BatchPlan.phase` 有 `mixed`，但 `PlanKey.phase` 只有 prefill/decode。建议首版明确走 eager，由 backend 按行组织执行；拆分方式及接口映射仍需落实，不能伪装成 decode graph。未支持的组合在状态更新前拒绝。2026-11-09 定：照 SGLang 的 `EagerRunner`，mixed 步当作 extend，走 bucket 0（eager），`PlanKey.phase` 记为 prefill（`core/contracts/plan.py` 的 `Backend`） |
| GDN speculative 部分接受 | M2d 根据所选 SGLang 路径确定中间状态保存或恢复重算方案、历史容量、tree/chain 能力及验收范围；实现前拒绝对应 verify 组合 |
| Backend 输入与绑定 | `forward(layer, *args)` 尚未表达 full attention/GDN 的参数语义。分别明确 tensor 顺序、shape/dtype、是否已经 normalization/scale/gate 变换，以及权重/状态绑定来源；在模型层与 backend 分别实现前落实。2026-11-09 已落实：`AttentionBackend.forward`、`GDNBackend.forward` |

~~当前 `core/contracts/plan.py` 的 docstring 仍将 prepare 描述为 host 工作、forward 描述为只在 graph 内运行~~：2026-11-09 已按 §8.4 改写，并补了 `Binding.generation` 的 stride 和提供者实例规则（§9.3）。

**验收要求：** 按 §5.5.8 的支持矩阵覆盖连续执行与 chunk/prefix 恢复后的输出及状态、verify 接受零个/部分/全部 token、eager/graph 等价性，以及配置切换前后的执行版本一致性。增加 A/B/C 对抗用例：只恢复 recurrent 而遗漏 conv、重复 normalization、提前覆盖下一步元数据、混用 A 的 prepare 与 B 的 graph、候选 capture 改写活跃状态、依赖失效后仍 replay。错误实现应被测试拒绝；stream 时序、capture 隔离和切换中的在途工作必须有真实 GPU 测试，不能仅靠 fake 平台通过就宣称正确。

## 9. plan 单元和 CUDA graph（@gaow0007 审）

交接见 `core/contracts/plan.py`。graph 的依赖表、重录迭代器和缓存组织留在 plan 实现内部。

### 9.1 plan

**执行范围（2026-11-09 讨论确认）：** plan 是给定有效配置和输入规格下的模型执行方案，可以走 eager，也可以持有 CUDA graph。首版采用完整模型 forward 的边界；CPU 参照、GPU eager 和 GPU graph 使用同一逻辑范围。

| 执行位置 | 工作 |
|---|---|
| worker，在 plan 执行前 | 校验并选择本步执行方案，安排 prefix/state restore，准备输入及 batch/backend 元数据 |
| `plan.run()` | embedding、全部模型层、final norm、LM head；各层内包含 full attention 的 KV 写入，以及 GDN 的 conv/recurrent 计算和状态更新 |
| worker 组织的后续处理 | 消费 logits，执行 sampling 或 speculative acceptance，安排快照及接受/拒绝后的状态处理；具体顺序遵守 §8.3 |

- 首版 graph 捕获完整模型 forward，暂不拆成分段 graph。采样、speculative acceptance、prefix restore、快照和拒绝后恢复在模型 graph 外；它们仍可在 GPU 上按 stream 依赖执行，不要求把 logits 拷回 CPU，也不意味着每步 CPU/GPU 同步。
- logits 写入 plan 绑定的输出 buffer。`plan.run()` 返回表示执行已提交，不代表 GPU 已完成或调用方取得永久独占的输出；同一步的后续 GPU 消费者按 step stream 顺序读取。下一次执行覆盖该 buffer 前，所有读取必须完成；跨 stream 消费须建立依赖，跨步保留须使用独立存储或明确的多 buffer 方案。输出生命周期与 graph 资源回收遵守 §9.2、§8.6。
- 该范围是实现与验收目标；当前 `Plan.run()` 的契约尚未说明 logits 的具体绑定交接，后续实现时补齐实际调用方所需接口，不在本次文档修改中预设新的公共类型。

**选择与分桶（2026-11-09 讨论确认）：** 对外保留简单的选择接口，内部区分有效配置、执行形状和依赖有效性；plan 是派生产物，依赖登记给 `core.deps`。

| 执行模式 | 首版策略 | 选择条件 |
|---|---|---|
| 普通 decode | 每个请求一个 token，按请求数分桶使用 graph | 同一有效配置内可用 `(phase, bucket)` 选择；还须通过绑定、solution 等依赖检查 |
| Prefill / chunked prefill | Eager，按实际 token 数执行 | 不把请求数当成 token 容量；`bucket=0` 表示 eager |
| Mixed batch | 明确支持后走 eager | 按 §8.7 落实阶段表示和分派；不得伪装成 decode 使用其 graph |
| Speculative verify | M2d 单独确定执行形状与支持路径 | 请求数相同不足以复用 graph；须考虑每请求 token 数、chain/tree 结构及算法要求。未验收的组合拒绝 |

- **Bucket 是捕获容量，不是当前真实请求数。** 例如 6 个普通 decode 请求可使用 bucket 8，另 2 行为 padding。真实请求数作为已验证的运行时元数据处理；§9.3 的 batch 大小按值依赖在这条路径中指捕获 bucket。若某个 kernel 仍把真实请求数作为捕获常量，则不能宣称支持该范围的 padding 复用。
- **Padding 必须对整个模型及状态操作安全。** padding 行不修改真实请求的 KV 或 GDN 状态，不进入有效采样结果；每个消费者使用的页号、槽号和长度都须合法。可使用实现明确支持的屏蔽或隔离 scratch，但不能默认槽号 0 或任意负数具有安全语义；共享 scratch 也不能引入并发写入冲突。
- **分桶策略可配置。** 初始默认在配置容量上限内取 2 的幂，同时支持显式 bucket 列表；启动或配置变更时校验容量、设备/backend 能力和资源预算。配置的 bucket 不等于已就绪的 graph，`PlanService.buckets` 交出的可选范围须反映当前有效配置下已就绪且依赖有效的 plan。
- **运行时选择容量合适的最小可用 bucket。** worker 确认配置和输入规格后，从已就绪、依赖有效且支持所需 padding 的 graph 中选择。调度侧给出的 bucket 不能绕过 worker 的有效性检查；选择、元数据准备与执行使用同一配置，遵守 §8.6。
- **缺失或过期 graph 不在步内重录。** 没有合适 graph 时，允许 eager 的配置走同版本、同语义且已验收的 eager 路径；要求 graph 的配置在本步状态更新前 fail-fast。backend/layout/精度等组合本身不兼容时直接拒绝，不能用 eager 回退绕过 §8.1。capture/补录由 §9.4 的安全窗口安排，不能藏进 `get` 或 `plan.run()`。
- **每套配置持有自己的 plan 集合。** A 与候选 B 可以各有 `decode/8`、`decode/16`，但其依赖和资源分开管理；worker 先确定本步配置，再在该集合内选择。配置版本不必加入公开 `PlanKey`，但必须参与内部集合隔离和有效性判断；准备 B 不得覆盖 A，切换按 §8.6 执行。这不要求同一 backend 单元的新旧实例并行存在。
- `(phase, bucket)` 的简单接口只覆盖已约定的模式。后续 verify、多 token 或其他影响捕获结构的输入规格，必须扩展实际选择规则和必要交接；不能仅复用相同键就认为形状兼容，也不提前引入未使用的公共字段。

**验收：** 覆盖真实 batch 等于 bucket、落在两桶之间、超过最大就绪桶，以及 bucket 已配置但尚未捕获、依赖已过期、禁止 eager 的情况。A/B/C 用例应能识别错误选择过小桶、padding 写入真实状态、把 B 的请求交给 A 的同名 plan，以及通过 eager 绕过显式不兼容配置。GPU 测试比较真实行的输出和状态，并确认其他请求状态不被 padding 改写。

### 9.2 录 graph

- capture 用的 stream 由平台提供（§10.4）。warm-up 和 capture 用同一个 stream，cuBLAS workspace 就不会随重载增长（探针 P3）。
- warm-up 只用 scratch 条目（§5.2）。
- **Graph 私有池与状态池分开（2026-11-09 讨论确认）。** 私有池保留 capture 期间分配的中间 tensor、输出等资源；权重、KV/GDN 状态池、预先分配的输入和 backend workspace 仍由各自提供者持有，不因被 graph 使用就自动进入私有池，也不要求每个 bucket 各复制一份。显式共享的外部资源仍须遵守 §8 的绑定、同步和生命周期约束。
- **同一配置的不同 bucket 允许共享 graph 私有池。** 前提是它们互不依赖对方输出、不并发执行，且下一次可能覆盖内存的 capture/replay 前，所有旧输出消费者已经完成；跨步保留的结果必须使用独立存储。正常步可按 `replay(bucket 8) → sampling/其他消费者 → replay(bucket 16)` 在 step stream 上排序。跨 stream 消费须有明确依赖；不能仅因使用同一个 step stream 就忽略其他消费者或 backend workspace。满足这些条件后，不要求 bucket 总按捕获顺序执行。
- **正在服务的 A 与准备中的 B 保持私有池隔离。** 候选 capture 不得复用或破坏旧配置仍需 replay 或回滚的资源。该隔离不替代外部 buffer 的保护，具体切换与回收遵守 §8.6、§9.4。独立池保留为基线和调试选项；共享池路径通过 GPU 验收前，不启用该优化。
- **释放以整个共享池的使用者为边界。** capture 内分配的输出也持有池资源，不能只释放 graph 而保留无用输出。只有相关在途工作及输出消费完成、池内所有 graph 和所需 tensor 引用均可释放时，才能回收整个池；卸载一个 bucket 不意味着共享池随之释放。
- **共享池验收：** 同配置下按不同顺序交替 bucket、重复 replay 同一 bucket，并覆盖实际 batch 小于 bucket 的情况；与独立池基线比较输出和 KV/GDN 状态。检查 logits 消费、跨步结果保留及旧 graph 释放时序；用 A/B/C 用例抓住输出提前覆盖、候选配置误用 active 池、仍有 graph 存活时回收共享资源等错误。记录共享/独立模式的显存占用，不能只凭数值测试通过就推断节省量。

上述共享条件参考 [PyTorch 的 graph memory management / sharing memory across captures](https://docs.pytorch.org/docs/2.15/notes/cuda.html#sharing-memory-across-captures)：互不依赖输出且不并发的 graph 可共享池，但 replay 可能覆盖其他 graph 的输出。P3 的输出覆盖现象用于约束生命周期，不能推导为“执行顺序变化时一律禁止共享”。

- 平台的 capture 原语**不在每次录制前清显存缓存**。`torch.cuda.graph` 每次进入都会调一次 `empty_cache()`，P9 里有一次花了 5.7 s。可以改用 `CUDAGraph.capture_begin` / `capture_end`，自己决定什么时候清（还没测过）。
- **capture 时，同一个进程里不能有别的 CUDA 工作**（P10）。默认的 `global` 模式只检查 CUDA 标记为 capture-unsafe 的 API 调用，不能当作保护。所以不在后台线程里录，只在 §9.4 的暂停窗口里录。

### 9.3 replay 前检查

**绑定版本检查（2026-11-09 讨论确认）：** 为支持 agxinfer 的资源变更和单元热更新，graph 记录并检查提供者身份及 binding generation。现有 `Binding.generation` 已声明该方向，本节补充其语义；这不是直接沿用 SGLang 的通用 replay 版本检查。在本次检查的 SGLang full/decode graph 路径中，主要由 runner 管理固定资源、判断执行规格并选择 graph；请求行复用所用的 `req_generation` 与这里的绑定版本不同。

**Generation 是绑定版本，不是内容版本。** 首版每个提供者（状态单元、backend、持久 batch、权重存储）使用一个粗粒度绑定计数器；它交出的任一相关 buffer 的绑定改变，就使该提供者旧绑定上的 graph 失效。先接受可能多重录一些 graph 的代价，不引入逐 tensor 的生产版本系统。

| 变化 | 绑定版本处理 |
|---|---|
| 原地址写入 KV、conv/recurrent state，或更新固定长度/页表/槽号 buffer 的内容 | 不递增；仍需遵守 stream 顺序和输入有效性约束 |
| 地址、shape、stride、dtype、layout 改变，或替换绑定中的 buffer 集合 | 递增；已有捕获不能按旧绑定继续执行 |
| 资源被释放或绑定被撤销 | 明确标为不可用并使旧版本失效；不能只递增后仍交出可用视图 |
| 原地替换权重内容且绑定结构不变 | 不仅因内容变化而递增；KV/GDN 状态语义是否失效另按状态兼容性处理 |

- **集中管理绑定变更。** 各提供者通过统一的内部替换/失效入口发布资源与版本，避免调用方直接改 buffer 而遗漏计数器。接口不要求一个通用的公共资源管理类。同一提供者实例内版本单调递增，回滚重新发布旧资源时也不把计数器退回，以免旧 graph 错误恢复有效。
- **按绑定检查 `(提供者实例身份, binding generation)`。** 复用已有单元 `InstanceId` 标识提供者加载实例；它的单元 generation 与 binding generation 是两个不同计数器。即使新旧实例的 binding generation 都是 1，实例身份也不同。依赖解析必须对应本步有效配置中的当前提供者，不能只持有旧 Binding 对象、反复检查旧对象自身。资源明确不可用时直接拒绝。
- **按值记录实际影响捕获的依赖。** 包括选中的 kernel/solution 及版本、相关代码实例、捕获使用的配置值、bucket 等固定执行规格和通信器身份/有效性。依赖既包括 capture 时观察到的值，也包括构造 backend、选择 solution 等准备阶段已经读取并缓存的相关值；不能仅靠 capture 期间的读取自动发现全部依赖。已验证的设备端运行时长度、page ID、slot ID 不因内容更新而成为按值失效项。
- 通信器仍是资源依赖图中的派生产物，拆除前停止依赖它的提交并等待在途使用结束，再拆 graph 和通信器。保存版本或对象引用不能替代实际 GPU 资源的保活。

**检查时机与失败分类：**

- worker 在本步状态更新前确定有效配置，选择 plan，并分别校验本步输入和 graph 依赖。有效 bucket 8 遇到 12 个请求属于输入不匹配，不是 graph 依赖失效；页号、槽号和状态边界校验也不能由 generation 比较替代。
- replay 前做轻量有效性确认。检查通过到提交期间不允许配置切换或绑定替换；`prepare` 不得偷偷换资源。提交后，相关资源保持有效直至 GPU 和必要消费者使用结束。通过 §8 的安全边界及 stream/event 顺序落实，不要求每步全设备同步。
- Graph 缺失、输入规格不匹配或依赖失效时，在尚未推进本步状态的条件下按 §9.1 重新选择 graph、同配置 eager，或 fail-fast。显式配置本身不兼容时不能通过 eager 绕过。若前置状态操作已经提交，或计算已开始，则不能直接重跑整步，须按 §8.6 判断恢复与继续条件。
- 失效 graph 从当前可选集合移除，不能提交。正常路径不构造详细诊断、不重复打印日志；失败时报告依赖提供者、失效类别、捕获值与当前值，并区分实例更换、绑定变更、资源不可用和按值依赖变化。

**实现与验收：**

- 正常路径比较提供者身份、generation 及已整理的按值依赖，以 CPU 元数据为主，不每步同步 GPU 或计算 tensor 内容哈希。目标仍是每步几微秒，但需在真实模型上测量；P6 逐 tensor 比对 55 项约 38 µs，不作为新方案的性能保证。
- 测试/调试模式同时比对实际地址、shape、stride、dtype、layout 和资源存活情况，用来发现“资源变了却未递增版本”。保留 fake 平台的详细检查，并补充真实 GPU 测试；当前 fake graph 的逐 buffer 检查不表示生产代数检查已经实现。
- A/B/C 用例覆盖：重载后绑定计数相同、同地址换 stride/layout、替换资源漏增版本、构造期配置依赖漏登记、仍检查旧提供者、检查后提前释放资源，以及回滚导致版本回退。正向用例确认固定 buffer 的内容更新不误报失效；GPU 测试确认被拒绝的 graph 没有执行或推进请求状态。
- 必须在提交前拦截：P1 已观察到 graph 使用已释放或换地址显存可能只读到垃圾；P5 已观察到模块卸载后旧 graph 仍可运行旧 kernel。不能把 CUDA 是否报错作为依赖有效性的判断依据。

### 9.4 重录

本节按 2026-11-09 的评审重写，规定重录与变更事务的交接；是实现目标，不表示当前框架已经提供 GPU 维护窗口或多 rank capture 编排。分片动作复用 [change.md](change.md) §4.6，最终切换遵守其变更事务；不得绕过 §8.6 的配置隔离和 §9.3 的依赖检查。

#### 9.4.1 先选择准备路径

重录前解析候选配置 B、受影响的 bucket、最终绑定、状态兼容/迁移要求及资源预算；旧配置 A 在允许的阶段继续服务。代码没有重载只是允许提前准备的必要条件，还必须满足可变资源隔离及候选绑定稳定。

| 路径 | 条件 | 准备与执行 |
|---|---|---|
| 分片准备 | 动作所在 plan 实例及其相关依赖不因本次变更被重载；候选资源可隔离；捕获所需最终绑定已确定；显存可容纳新旧资源 | Prepare 中分片构建候选 graph，窗口之间继续服务 A；最终暂停后复核并安装 B |
| 暂停期构建 | 需要原地重载/修改资源，最终绑定要到 Execute 才确定，或不能提前容纳候选资源 | Prepare 只做可提前完成的校验、规划或独立准备；依赖最终绑定的分配/capture 在 Pause 后的 Execute 内完成 |
| 拒绝或等待外部资源条件改变 | 无可行的显存/停顿预算，状态无安全迁移方案，或无法满足通信组协调条件 | 在修改 active 资源前 fail-fast；不得先 OOM 或破坏旧状态，再假定可以回退 |

- 只改分桶、在已注册 solution 间切换、通过新增单元提供 kernel，是分片准备的候选场景，不是无条件保证。场景 ④（`test_4_a_new_kernel_rebuilds_only_the_graph_that_uses_it`）验证新增 kernel 的框架流程，不替代真实 GPU 的资源隔离测试。候选值通过重录动作参数传入，不能为准备 B 提前修改 A 的有效配置。
- 原地改 `core.plan`、backend、模型、层、`core.kernels`、`worker.batch` 或相关共享接口，可能使动作实例被重载。当前 `Participant._prepare_or_defer` 按实际受影响单元推迟准备，并在 Execute 的新实例上一次推进完迭代器，报告 `change.prepare_one_shot`。因此“prepare 返回迭代器”不保证这类变更仍能在 serving 间隙推进。
- **资源动作的顺序必须显式规划。** 当前框架对被重载动作的 prepare 在 action execute 之前推进；若状态池的最终绑定要由另一 action 的 execute 才产生，不能在这个延迟 prepare 中提前 capture。重录 action 应在依赖动作完成后构建，或调整必要的动作交接，并在 Execute 成功前完成校验。不能假定框架会自动按 GPU 资源依赖重新排序。
- 同一个单元的新旧实例并行服务、在其他进程 capture 后迁入 graph 均不在 v0 范围内。独立候选 buffer/graph 不等同于同一 backend 的双实例热更新。

#### 9.4.2 分片窗口与多 rank 协调

- 重录 action 的 `prepare(args)` 可以返回迭代器，每次推进最多构建一个 bucket。窗口内先停止新的 GPU 提交，完成在途 step、结果收集及相关消费者/辅助 stream 工作，再在平台 capture stream 上 warm-up、capture 和验证；所有可能写状态的试跑使用 scratch。窗口结束后恢复 A，一步 overlap 暂时中断，不承诺无延迟抖动。
- **本地安全点不是自动完成的 GPU 暂停。** plan/worker 的实现负责进入、退出维护窗口；当前 participant 只是推进迭代器。恢复 serving 前须保证捕获结束、active 元数据仍完整或已按 A 重新准备、在途资源未被候选改写。
- 同一候选配置内 bucket 按 §9.2 可共享私有池，候选 B 与 active A 的池隔离。每片绑定快照与候选配置关联；准备过程中依赖变化时，该片失效，不能继续累计为 ready。对候选的验证使用其预期配置/绑定，不能为让检查通过而切换 active 配置，也不能无条件绕过依赖检查。
- **包含 collective 的 capture 按整个通信组协调。** 各 rank 必须完成同一执行边界的在途工作，按共同的 bucket、warm-up/capture/验证顺序进入窗口，确认完成后才能恢复 serving。不能让一个 rank 进入候选 warm-up，其他 rank 继续提交旧 step 的 collective。仅在各 rank 分别调用同步或 barrier，不足以替代停止提交和顺序协调。
- 当前 `_advance()` 在各参与进程独立推进，不提供上述组级保证。该机制实现并通过故障测试前，分片重录能力只承诺已验收的 TP1 路径；多 rank 只能使用已实现组级顺序及失败处理的统一暂停重录路径，否则拒绝该变更。统一暂停后也不能假定各 action 的 collective 顺序自然一致。
- 通信组须有一致的窗口标识与退出条件；任一 rank 未就绪、失败或超时，不能让部分 rank 恢复 A、部分 rank 留在候选计算。进入统一撤销或恢复流程；设备/通信器已不可用时不得强行继续 serving。

#### 9.4.3 最终复核、安装与生效

| 事务阶段 | Plan/worker 的责任 |
|---|---|
| Prepare | 校验并按选定路径准备候选资源，记录已就绪与延迟构建的 bucket；明确所需资源预算和撤销方案，不改变正在服务的配置 |
| Drain / Pause | 按状态变更需要排空请求；按协议边界 quiesce，完成在途计算及结果消费，停止新提交。排空请求与 GPU 静止是不同条件 |
| Execute | 完成重载、资源动作和状态迁移；在最终绑定确定后构建缺失 graph，并复核候选依赖、输入规格及状态兼容性。安装 B 的配置/绑定/plan，但继续暂停，保留可执行的撤销方案 |
| Commit | 接受已持久化的提交决定，不再回滚；必要 capture、迁移及正确性校验不得留到此阶段。按协议和资源存活条件安排旧资源回收 |
| Resume | 按变更协议恢复，开始提交 B 的 step；多 rank 使用一致配置和执行方案 |

- Execute 中复核 §9.3 的提供者身份、binding generation、solution、捕获配置与通信器。Prepare 时的 ready 不保证切换时仍有效；失效则在暂停期按预算重建或使 Execute 失败。不得先恢复服务，再让第一次 replay 才发现新 graph 无效。
- 当前 participant 的 commit hook 异常主要记诊断，不作为提交前失败返回；因此关键构建/验证必须在 Execute 成功前完成。Execute 安装新 plan 不代表可以提前 Resume 或释放仍需要撤销的资源。
- 默认允许在已验收的同配置 eager 可用时，只在切换前构建常用 bucket；B 生效后未就绪桶走 B eager，再用维护窗口补录。要求 graph 的配置须在切换前具备所需 graph；缺失不能偷偷回退 A。Eager/graph 的输出与状态等价性按实际 kernel 组合验收，不能把“逐位一致”当成所有路径天然成立的事实。
- B 生效后的补录是该配置内部的派生缓存维护，不修改模型语义或有效配置；仍受 capture 隔离、依赖、内存和窗口预算约束。配置再次变化后，不发布基于旧依赖完成的补录结果。

#### 9.4.4 显存预算、失败恢复与回收

- 显存预算按实际存活资源计算：可共享且未复制的权重/状态池只计一次；分别计入 A/B 的 graph 私有池、独立 metadata/workspace/scratch、需要迁移的第二套状态、通信器及构建临时峰值。同配置共享池不能把每桶占用简单相加，新旧配置隔离也不意味着所有模型状态都翻倍。
- **排空不自动释放显存。** 请求结束可能只把页还给预分配池；旧 graph、输出引用和通信器仍可驻留。预算必须列明哪些资源会在什么阶段释放，以及释放是否影响撤销。当前协调者先 Prepare 再 Drain/Pause，故显存不足路径必须在 Prepare 阶段选择延迟构建，不能先分配完整 B 再等后续排空解决 OOM。
- 若必须先回收旧 graph 才有空间构建 B，应在暂停、无在途使用后执行，并预先验证失败时能重建 A 的 graph 或使用合法的 A eager 恢复。状态资源被迁移或释放时，还须有可行的状态撤销/重算方案及其资源预算。未提供可执行恢复方案就拒绝该路径；不能同时承诺释放旧资源和立即复用旧 graph。
- 候选准备失败且 active 完整、CUDA 上下文仍可用时，清理候选并保留 A。Execute 失败按 Abort 恢复配置、状态及可用执行方案；恢复绑定遵守 §9.3 的单调版本规则，不能把旧 graph 指针放回去就视为回滚成功。恢复失败进入 recovery-required，不能报告 aborted 后继续推进请求。
- 提交决定形成后不再退回 A；Commit/Resume 确认失败或设备故障走框架恢复流程。捕获异常后的 CUDA 上下文、通信器及外部资源可用性必须确认；不可用时重启/恢复，不在未知状态下重试模型步。
- 清理遵守共享池整体生命周期：等待 graph、输出、外部消费者及通信操作结束，且事务不再需要相关资源后才能回收。候选失败清理不得释放 active 的外部绑定，也不得全局清理仍被其他 graph 使用的 workspace。

#### 9.4.5 停顿预算与验收

- 预算针对**整个维护窗口**，包括等待在途工作、多个准备动作、warm-up、capture、验证及恢复元数据。当前 `_advance()` 每次推进所有未完成准备各一片，因此多个迭代器会累加停顿；要达到“一窗口最多一个 bucket”的策略，须由维护窗口编排限制实际 capture 数量，不能仅依靠迭代器数量。
- 提供可配置的窗口频率、每次推进量和准备资源上限；根据测量选择常用 bucket 的顺序。单次 capture 不能假设可中途抢占；预计超出预算时推迟、减少本次工作或采用暂停期路径，实际超预算则记录并调整后续窗口。协调者 prepare 超时、组级失败处理和心跳也须计入长准备过程，不能用延长租约掩盖无界阻塞。
- dry-run 报告所选路径、预计 bucket 数、资源峰值、停顿和回退条件；运行日志记录每个窗口实测值、就绪覆盖及失败原因。估计来自对应设备/kernel 的测量，无法估计的部分明确标为未知。
- P9 的历史测量：Qwen3.8-27B 尺寸、36 个 bucket，平时一步约 30.6 ms；单窗口约 0.2 s（等待约 30 ms、两次 warm-up 约 110 ms、capture 约 47 ms），合计约 6.6 s，一次性重录约 6.9 s。这些不是当前默认分桶、其他设备或新 kernel 的延迟保证。每个版本只 warm-up 一次、跳过部分 shape 的 warm-up 均为待验证优化。
- GPU 验收覆盖：在途 step 未完成时进入窗口；候选准备期间 A 继续正确服务；Execute 更换绑定使候选失效；部分 bucket 就绪后切换并走 B eager；多 rank 延迟/失败和 collective 顺序；候选 OOM 与资源回收；Execute 失败后恢复；Commit/Resume 边界；共享池下补录和旧输出消费。数值、状态和时序都要检查，不能只确认 graph 可 replay。
- A/B/C 对抗用例应拒绝：只同步本 rank 就开始 collective capture、准备时改写 active buffer、安装已过期候选、Execute 后提前 serving、关键 capture 放到 commit hook、排空后仍假定显存已释放、回滚恢复旧指针却忽略新 generation，以及多准备动作突破窗口策略。框架 fake 场景验证事务接缝，真实 GPU 测试验证内存和执行顺序；两者不能互相替代。

## 10. 平台 v2（@gaow0007 审）

### 10.1 三个平台

| 平台 | 用途 | 有没有计算 | graph |
|---|---|---|---|
| `fake`（已有） | 测引擎的机制：变更、重载、故障 | 没有，只有按名字调用的假 kernel | 记录依赖，replay 时检查 |
| `torch_eager`（PR #24 已合入 master，2026-11-08） | 在 CPU 或进程自己的 CUDA 卡上跑真模型语义：M2a、CI 里的单元测试和五组对比、SM100 上的 eager | torch 张量、参照 kernel | 不录 graph（`graph()` 报错），见 platform.md §2.3 |
| `cuda_sm100`（新，@gaow0007） | SM100 | torch.cuda | 真 CUDA graph |

- `torch_eager` 和 `cuda_sm100` 跑同一套平台契约测试。
- `torch_eager` 上，“graph 对 eager”自然完全一致，这组对比只在 GPU 层有意义。

### 10.2 buffer

**2026-11-09 讨论确认：`tensor` 交出共享视图，不转移底层资源所有权。** 在 torch 平台上，Buffer 封装实际分配的 tensor/storage；通过 `tensor` 交出的对象与之共享存储，不隐式复制，类型在共享契约中保持不透明。Fake 平台仍无真实 tensor，用元数据和受控操作验证契约。

- **所有者负责分配与释放。** 状态、权重或 backend 等提供者持有相应 Buffer，借用者按职责读写内容；取得 tensor 不意味着可以释放底层资源。Graph 输出的所有权和消费顺序按 §9.1、§9.2 处理。
- **借用者不得改变已发布绑定的结构。** 不得对共享对象执行 resize、替换 storage、原地改写其 shape/stride 元数据或自行释放。可在约定范围内创建切片/视图，但不能越界、隐式扩大写权限或将其当成独立分配释放。需要持久独占副本时显式复制，并单独管理其生命周期。
- **内容读写遵守调用职责与 stream 顺序。** 同地址更新内容不递增 binding generation；权重视图通常只读，只有授权的 refit 路径更新。可写状态也不能任意覆盖共享 prefix、其他请求或在途消费者正在读取的内容。只读/可写是调用契约，不假定普通 torch tensor 自动提供权限隔离。
- **结构信息必须反映实际视图。** 绑定交接能够核对 device、dtype、首元素地址、shape、stride、相对底层 storage 的 offset，以及实际可访问的存储范围。Stride/offset 的单位须明确；非连续视图不得当成连续 tensor，也不能仅用逻辑元素总字节数推断其实际覆盖范围。只补实际消费者需要的字段和检查，不增加通用 tensor 镜像。
- **结构变更由所有者集中发布。** 替换 buffer、改变地址/shape/stride/dtype/layout 或撤销绑定时，按 §9.3 更新 generation 并使受影响 graph 失效；需要时按 §9.4 迁移、重录和切换。调用方不得以修改共享 tensor 的方式绕过这一入口。
- **释放遵守 §10.6。** Tensor/view、可再次执行的 graph 和在途 GPU 工作都属于资源使用者；丢掉 Buffer 的一个 Python 引用不代表可以释放，持有失效视图也不授权继续使用。所有者负责阻止新使用，并在相关使用结束后完成回收。
- 状态池启动时按显存预算确定容量，不要求在线扩容。验收覆盖共享视图写入可见性、切片别名及 stride/offset、结构变更触发 graph 失效、无权释放/越界访问、在途使用时不提前回收；不要求 fake 自动拦截任意 torch/CUDA 操作，真实路径另做 GPU 验证。

### 10.4 stream

**2026-11-09 讨论确认：平台持有稳定 stream，worker 决定执行顺序。** 同一进程、设备和用途重复获取时复用相同 stream；单元重载不重建 stream。首版不引入通用 stream scheduler。

| 用途 | 工作 | 首版约定 |
|---|---|---|
| `step` | batch/backend 元数据更新、状态复制、模型执行、采样及相关结果处理 | 正常步默认路径，按 §8.4 的依赖顺序提交；普通状态复制也先用此 stream |
| `capture` | warm-up、capture、候选验证 | 只在 §9.4 维护窗口使用，进入前完成相关在途工作；warm-up 与 capture 复用该 stream |
| `copy` | 需要独立重叠的传输 | 按实际需求启用，不默认把所有 copy 移过去；显式声明生产者/消费者之间的等待关系 |

- 现有 `Platform.stream(device)` 需按实际调用扩展用途选择、执行上下文及必要的同步交接。辅助 stream 的使用者必须保证 event/等待顺序和 buffer 保活，worker 的 quiesce 覆盖它提交的相关工作；取得 stream 句柄本身不代表已经进入该执行上下文。
- 单元不自行创建 stream。manifest 对直接创建的检查作为 lint，不视为能够拦截别名、包装函数或第三方库的完整保证；第三方内部 stream 由适配层声明并通过时序和资源测试验收。
- 稳定复用 stream 用于减少重载时重复创建 cuBLAS workspace 等资源。P3 观察到相关显存增长约 270 MB，是特定探针结果；固定 stream 数量不能保证其他 workspace、graph 池或第三方缓存永不增长。验收须测重复重载后的资源占用，而非只数 stream。

### 10.5 graph

**平台提供底层原语，plan 管理执行策略。** 按 §9 落实，不在平台内重复实现配置选择或另一套 plan 缓存。

| 平台负责 | Plan 负责 |
|---|---|
| capture、replay、底层 graph 销毁及失败报告 | 执行范围、bucket、eager/graph 选择及重录 |
| 提供 graph 私有池创建、共享和释放所需能力 | 同配置 bucket 的共享关系、active/candidate 隔离及资源预算 |
| 按调用约定执行底层资源操作，支持依赖检查所需的观测 | 依赖登记、有效性判断、候选准备和变更事务交接 |

- capture 失败不得发布 ready graph，清理候选时不得影响 active 资源；错误交给 plan/worker 按 §8.6、§9.4 处理。已开始的 GPU 执行失败不能由平台悄悄重跑。
- 同配置 bucket 可按 §9.2 共享私有池，候选 B 与 active A 隔离。Graph 停用、底层对象销毁和共享池最终回收是不同事件；回收依赖最后一个使用者及在途工作结束，不只依赖 Python GC 时机。
- replay 的执行入口在提交前落实 §9.3 的提供者身份、binding generation 和按值依赖检查。保留 fake 平台的详细地址/结构检查，用于发现漏登记或漏增版本；CPU 模拟验证接口和状态语义，不能代替真实 CUDA capture、异步时序和共享池验收。
- 只增加上述实际交接需要的底层接口，内部池组织和依赖表留在实现中；当前 `GraphRecorder`、`Graph` 与 fake 实现仍需后续代码落实。

### 10.6 收尾和清理

**先停止使用，再释放资源。** Worker 的 quiesce 按以下顺序完成：停止新的相关提交，等待在途 GPU 工作和输出消费者完成，收集结果并更新已完成 step 状态。需要卸载或回收时，再停用相关 plan，按依赖顺序释放 graph、输出和其他资源。单纯 quiesce 不要求销毁 graph；配置切换的撤销能力与回收时机继续遵守 §9.4。

- `platform.synchronize(device)` 只提供设备等待能力，不负责停止新提交、收集结果或切换配置。首版允许在维护窗口使用设备级同步以简化正确性；正常 serving 不增加每步全设备同步。涉及多个设备/通信组时仍需 §9.4 的共同边界，不能以本地同步替代组级协调。
- buffer 释放同时检查持有它的 tensor/view、可再次执行的 graph 和已提交 GPU 工作；没有 Python view 引用不代表已经可以释放。跨 stream 的保活和结果交接由实际使用者负责，资源所有者执行最终释放。
- 清理、同步或撤销失败必须报告，不能把部分释放视为卸载成功或立即复用相关资源；按变更协议进入回滚或恢复。共享池只有在全部相关 graph、输出及在途使用结束后才能回收。
- **普通单元卸载不清理全局 cuBLAS workspace。** P3 使用的 `_cuda_clearCublasWorkspaces()` 影响进程级缓存，其他 graph 或在途工作可能仍引用其地址。若确需调用，应由进程级生命周期管理在确认无使用者后处理，并针对锁定的依赖版本验证；不能成为每次卸载的默认动作。

### 10.7 外部注册与撤销

- 外部注册可能在模块卸载后仍存活，P2 已观察到 `torch.library` 重复注册问题。每项注册明确唯一所有者、作用域和注销顺序；不能假定删除模块就撤销注册。
- **可可靠撤销的注册**通过 `ctx.effect` 管理（[unit-runtime.md](unit-runtime.md) §5）。注销前停止相关执行、完成在途调用并停用依赖 graph；注销本身不等于旧 graph 已经失效。提前撤销、正常卸载和失败回滚不得重复破坏同一资源。
- **Setup 的部分失败由 setup 自行清理。** 当前 `ctx.effect` 先执行 `setup()`，成功取得 undo 后才登记。因此 setup 注册了一部分就抛异常时，框架尚无撤销函数；setup 必须清理已完成操作，或把可独立撤销的步骤分别登记。不能仅以调用了 `ctx.effect` 就宣称异常路径安全。
- **无法可靠撤销的注册**按进程级生命周期管理，需要替换时重启相关进程；在变更规划阶段识别，不能等热重载失败后继续使用部分注册状态。
- P2 使用 `Library._destroy()`，属于私有 API 探针。实现须针对锁定版本验证具体撤销方式，不将该方法或所有 `torch.library` 注册都视为无条件支持热卸载。
- 验收覆盖重复加载/卸载、setup 中途失败、提前撤销、回滚及 graph 存活时的注销限制；A/B/C 用例植入漏撤销和部分注册泄漏，检查后续重新加载及调用语义，不能只检查无异常。

### 10.8 代码重载

**不可变快照解决源码身份，代码/solution 依赖解决实际执行版本。** Python 模块加载成功不代表已有 graph 自动运行新 kernel。

| 对象 | 更新要求 |
|---|---|
| Python 源码 | 从对应身份的不可变快照加载，不原地改写已物化源码 |
| Kernel/solution 及编译产物 | 可追溯到实际代码、相关编译选项、目标设备及依赖版本；验证变化时能选中对应产物 |
| 已捕获 graph | 按 §9.3 检查代码实例、solution 等依赖；失效后按 §9.4 重录和切换 |

- P7 观察到原地改写同一 `.py`、同一秒且长度不变时可能命中旧字节码。现有机制每个身份使用独立快照目录，节点代理设置 `PYTHONDONTWRITEBYTECODE=1`，change-set/blob 排除 `__pycache__`；控制面生成 `.pyc` 不改变已物化源码。
- 节点代理和控制面按摘要命名快照目录，先写临时目录再原子改名，已存在则不重写；节点代理还设置只读。保留 `tests/framework/cluster/test_cluster.py::test_a_snapshot_is_written_once_and_never_rewritten` 的验收。这里的不可变性约束针对源码内容，不以禁用全部编译缓存代替身份管理。
- JIT/编译缓存优先使用实际工具链支持且已验证的身份机制；自定义缓存键必须覆盖影响产物的输入。不能只凭路径或函数名判断新旧实现，也不默认每次重载清空全局缓存。相关库私有行为须按项目锁定版本验证。
- 区分两类失效：代码/solution 变化使相关 graph 失效；计算或权重数值语义变化还可能使 KV/GDN 状态、prefix cache 失效。后者须迁移、重新 prefill 或排空，不能只重录 graph 后继续使用旧状态。
- 验收从新快照加载后实际执行的新 kernel/solution 及输出语义，并拒绝旧 graph；另覆盖不改变数值语义的实现替换和改变语义时的状态失效。只检查 import 成功、模块路径变化或 graph 能 replay 不足以证明更新完成。

## 11. 测试和工具链

### 11.1 测试分层

| 层 | 位置 | 平台 | 谁跑 |
|---|---|---|---|
| 单元的 CPU 测试 | `tests/units/<单元路径>/` | `torch_eager`、`fake` | CI |
| 引擎级 | `tests/scenarios/` | `fake`，M2a 以后加 `torch_eager` | CI |
| GPU | `tests/gpu/` | `cuda_sm100` | 在 SM100 上手动跑，像 formal 一样由 `addopts` 排除；@gaow0007 跑自己那几块，Devin 跑集成 |

- 平台契约测试和 kernel 契约测试（solution 对参照实现）对每个平台、每个 solution 各跑一遍，用参数化，不用 skip。
- 五组对比：CPU 层用参照 kernel 跑，GPU 层用小尺寸配置加真 kernel 跑（只用和 batch 怎么拼无关的 solution）。

### 11.2 工具链

- GPU 一档是 torch 2.11 / Triton 3.6，fla、flashinfer 钉死版本：fla 0.6.0 在 Triton 3.3.1 上做 chunk prefill 会挂住（P5）。
- torch 不进默认依赖。用 uv 的两个互斥 extra：`cpu`（CPU 版 torch，CI 装）和 `gpu`（CUDA 版 torch、flashinfer、fla）。
- 不用 torch 的框架测试，在两种环境下都能跑。

## 12. 纸面包络测试

用本文的接口检查三样东西装不装得下。表达不了的写进包络失效日志。

### 12.1 一步重叠

装得下。需要 §4.4 的全部九条，加上框架的 `quiesce()`（PR #18）。

### 12.2 LoRA（v0 不实现）

| 需要什么 | 用什么表达 | 结论 |
|---|---|---|
| 每行用哪个 adapter | LoRA 单元在 `StepPlan` 里贡献自己的分段（P2） | 装得下 |
| adapter 权重放在显存的固定槽位里，请求共用 | 一种 `slot` 寿命的状态。账本内部的状态池按请求数给槽位记引用计数；“请求要的 adapter 已在槽位里”和 KV 一起参与联合准入 | 装得下 |
| 分段的 LoRA GEMM（sgmv 一类） | kernel 目录里的一个 definition 加 solution | 装得下 |
| graph | 槽号 buffer 按地址，槽位里的权重按地址。载入 adapter 是同地址换内容，不用重录 | 装得下 |
| 新加一个 adapter | change-set 里加一个产物，身份跟着变。载入槽位属于数据面的缓存，不是变更 | 装得下 |
| 前缀缓存 | 同一段 token 在不同 adapter 下的 KV 不同，所以块哈希必须把会改变 KV 数值的东西（adapter id）也算进去 | **要补**：块哈希的输入要开放给单元贡献（失效日志第 2 条） |

### 12.3 DS 1017（v0.1）

在 modules.html 思想实验一（16 步）的基础上，用本文的接口重新核对一遍：
- 压缩 KV：每层压缩比不同，就是每层的 `tokens_per_item` 不同。
- indexer K：另一种状态。
- 跨层借用 KV：到 v0.1 实现时补充实际需要的声明。
- 每序列一个 state slot，分可持久和可丢弃两区：两种 `slot` 状态。
- nextn 层的 KV：再一层声明。
- Engram host 大表：到该单元实际加入时定义需要交接的放置数据。
- 专家放置表和负载计数：权重存储的 `relocate`，加一种状态。
- DP8 + EP8：每个 rank 一份 StepPlan（`OpenRecord.rank`）；账本按调度域分；同一个 EP 组的所有 rank 每步都要走，没请求的跑空 batch。
- indexer 在 GPU 上选块：在 graph 内做。

这些是后续实现的扩展方向；不为纸面覆盖预先固化所有字段，允许同一个 agent 同时更新交接定义和使用方。

### 12.4 包络失效日志

1. **已解决：动作名开放（PR #21，2026-11-08 已合入 master）。** `capabilities.actions` 使用 `ActionName = str`，状态单元和逻辑单元都可以声明自己的动作名；框架只校验名称形式，具体动作由单元注册和执行。调度逻辑可声明 `drain`，新增算法动作无需修改框架枚举。`StateRequest` 仍只作用于状态单元；自动排空的依赖推导与调度动作实现继续按 `state-changes.md` 推进，#21 只解除声明限制。
2. **块哈希的输入要开放。** 前缀复用的块哈希默认只算 token。会改变 KV 数值的东西（LoRA adapter、多模态输入的哈希）要能由单元贡献进来。实现前缀复用（M2d）时一起做。

## 13. 探针报告的建议怎么落地

| 报告的建议（按合并时的编号） | 落到哪里 |
|---|---|
| 1. replay 前检查改成比对绑定代数；代码版本是按值依赖 | §5.2、§9.3、§10.5 |
| 2. stream 由平台按用途统一管理；单元不建 stream；卸载时不清 cuBLAS workspace | §9.2、§10.4、§10.6 |
| 3. `torch.library` 走 `ctx.effect` | §10.7 |
| 4. graph 连输出一起释放；不和旧 graph 共享显存池；capture 不和服务并行，只在暂停窗口里录；capture 前不清显存缓存 | §9.2、§9.4 |
| 6. scratch 条目 | §5.2 |
| 7. backend 按 sglang 分工 | §8 |
| 8. 快照路径的内容写好以后不再改变 | §10.8 |
| 9. Qwen3.8 的状态规模、MTP | §5.4 |
| 10. 工具链 | §11.2 |
| 11. 进程内改 TP | **v0 的默认做法不变**，TP 变更继续重启 rank 进程（modules.html 里 `dist.groups` 的“改并行配置：重建 rank 进程”）。P11 用真实权重测过：进程内经 NVLink 补齐权重约 0.7 s，重启进程到权重就位要 35 s，值得以后去争取。但 KV 和 GDN 状态按新切法怎么处理、中途失败怎么恢复、怎么扩大 world，都还没测。决定 TP 变更要重启的是控制面单元 `core.deps`，以后改不用重启引擎。现在先记进规格的只有：通信器是一类派生产物，及其显存（§9.3、§9.4） |
| 12. 换了中间层的权重，KV、GDN 状态和前缀缓存都要失效，在途请求按 token 历史重新 prefill | 和 v0-plan 4.5 的默认推导一致。只有能证明不影响已有状态的权重（例如输出头）才只做 refit、不排空。“退回排队、重新 prefill”是排空时持有请求的单元可以选的处理方式（[state-changes.md](state-changes.md) §5） |

## 14. 和已有文档的关系

- v0-plan 第 128 行原来把异步调度列为“只影响快慢的优化，v0 一律不做”；5.12 写的是“同步调度，不做异步调度”。现在改成：契约装得下一步重叠，M2d 做一个最简版（§4.4）。这两处已经改了。
- roadmap 阶段 2 的验收，从四组对比改成五组。
- 本文定稿以后，[platform.md](platform.md) §2.1 和 §3 要按 §10 改，并且照 §10 改假平台的实现、补契约测试。这是下一个 PR。

## 15. 待定

1. §9.4 里还没测过、还没做的几件事：capture 前不清显存缓存、每个版本只 warm-up 一次、GPU 正忙时停下并切换；dry-run 怎么报重录造成的停顿。不能分片时走哪条路径，按 §9.4.1 的表选。
2. ~~`torch_eager` 平台上的 graph~~：已定（2026-11-08），`torch_eager` 不录 graph；“录制时写死 Python 标量”这类错误由 `fake` 和 GPU 层抓。
3. 前端和调度之间的流在 M2a 用每步记录流。前端以后要不要拆成多个进程，到时再定。
4. ~~采样放在 graph 里还是 graph 外~~：已定（2026-11-09，§9.1），首版采样在模型 graph 外，采到的 token 留在显存里给 `from_previous` 用。以后要不要融进 graph，另行评估。
5. 投机的记录：草稿放在哪、验证步的 `num_tokens` 怎么数。M2d 定，是否新增分段或版本按 §3 判断（§4.2）。
