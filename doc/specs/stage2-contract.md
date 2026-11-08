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
| @wutianyuan1 和 agent 们 | 其余全部：前端、调度、账本、请求表、每步执行循环、持久 batch、模型、层、kernel 目录和参照实现、状态单元、权重存储、`platforms.torch_cpu`、控制面单元的新规则、集成测试和全模型对齐 | Devin 在 SM100 上做集成验证和补测 |
| 本文和 `engine/core/contracts/` | 两边共有 | 团队 PR 由双方 review；同一 RSI agent 可一起修改接口与全部使用者，不增加运行时审批 |

分工的边界划在**可用 CPU 参照实现验证的接口**上：
- 平台有 `torch_cpu` 和 `fake`；
- kernel 有 PyTorch 参照实现；
- graph 有 eager 路径。

所以 @wutianyuan1 这边不用等 GPU 部分，就能把整条链路跑通；@gaow0007 按契约和契约测试实现，不需要了解引擎全局。这几块的契约测试由 @wutianyuan1 这边先写好。临时固定输出等替身只放测试夹具，真实实现可用后删除；参照 kernel 和真实 CPU eager 执行则保留用于数值对齐。

### 1.3 里程碑

| 里程碑 | 内容 | 主要靠谁 |
|---|---|---|
| M2a | 小尺寸、随机权重的 Qwen3.8 结构（例如 4 层：3 层 GDN + 1 层 full attention），参照 kernel，eager，在 `torch_cpu` 上把前端 → 调度 → worker → 前端整条链路跑通，和同尺寸的 HF 参照逐 token 对齐 | @wutianyuan1 |
| M2b | SM100 上的 TP1，换成真 kernel，仍然 eager，全模型和 HF 参照在容差内对齐 | 两边集成 |
| M2c | decode graph：graph 和 eager 逐位一致；能分片的变更在准备阶段按分桶切片重录，不能分片的按 §9.4 处理 | @gaow0007 为主 |
| M2d | 前缀复用、MTP 投机、流式输出；五组对比；一步重叠；一种多 rank 配置 | @wutianyuan1 为主 |

验收沿用 roadmap，另加一组对比（§4.4）：
- 全模型和 HF 参照在容差内对齐；
- 快速测试层里，“批量对单条”“前缀命中对不命中”“开投机对不开”“decode graph 对 eager”“开重叠对不开”五组对比完全一致；
- 能服务流式请求。

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
| | `models.qwen38` | 逻辑 | `model@1` | `layers.*`、`state.weights@1` | |
| | `layers.*`（linear、embedding、lm_head、norm、rope、attn 外壳、gdn 外壳、sampler） | 逻辑 | `layers.<名字>@1` | `kernels.catalog@1` | |
| | `core.kernels` | 逻辑 | `kernels.catalog@1` | | |
| | `kernels.reference` | 逻辑 | 往目录里登记参照实现 | | |
| | `kernels.flashinfer`、`kernels.fla` | 逻辑 | 往目录里登记 solution | | ✓ |
| | `attn.backend.flashinfer`、`gdn.backend.fla` | 逻辑 | `attn.backend.*@1`、`gdn.backend.*@1` | 状态单元、`worker.batch@1` | ✓ |
| | `core.state.holder` | 状态 | `state.holder@1`（显存预算、VMM 预留） | | |
| | `state.kv_paged`、`state.gdn_slots` | 状态 | `state.kv.paged@1`、`state.gdn.slots@1` | `state.holder@1` | 布局 ✓ |
| | `core.state.weights` | 状态 | `state.weights@1` | `state.holder@1` | |
| | `core.dist`、`dist.groups` | 逻辑、状态 | M2d 才要 | | |
| 控制面 | 已有的四个单元 | | | | |

- 控制面单元要加的规则：
  - `core.deps`：重录 graph 的推导（graph 的依赖登记成派生产物，依赖变了就重录：能分片的在准备阶段，不能的在暂停期内，§9.4），以及 grow；dry-run 报出重录造成的停顿；
  - `core.manifest`：单元不许自己建 stream（§10.4）。
- 新平台 `platforms.torch_cpu` 和 `platforms.cuda_sm100` 都属于框架本体，要加进 `engine/framework.toml`。

## 3. 最小共享交接面（2026-11-08 更新）

按发起人的最新决定，共享定义只描述实际跨单元、跨进程交接的数据和必要调用。内部对象布局、算法、队列、缓存索引和辅助函数归实现所有。不能因为一个功能需要类型提示，就把整套业务对象搬进 contracts。

- 新增算法、换 kernel、改调度策略，沿用现有交接方式时直接改实现。
- 交接内容或语义确实改变时，才更新共享定义；单个 RSI agent 可以同时修改接口及全部使用者。
- 所有使用者在同一次变更中更新时，可以直接修改定义。需要不同版本并存时再新增分段或版本；不要求每个功能都新增文件、分段或 `@2`。
- 两位负责人审核是当前团队的 PR 协作流程，不是 RSI 运行时的人类审批条件。`apply` 仍按依赖、暂停和变更协议执行。
- `__init__.py` 不重新导出各模块。模块按真实交接边界隔离，避免无关重载；`REACHES` 测试记录实际依赖。
- 修改前后用依赖分析确认影响范围。没有运行实例时可检查 ImportGraph；有实例时用 dry-run。它是工程检查，不是审批。

当前保留：

| 模块 | 实际交接 |
|---|---|
| `requests`、`sampling`、`columns` | 前端、调度器和 worker 使用的消息与采样数据 |
| `records` | 调度器发给 worker 的步计划和结果 |
| `state`、`storage`、`tensor` | 模型向存储/账本提供的当前状态声明，以及 buffer 交接 |
| `kernels`、`model` | kernel 注册和选择，以及模型向 plan/权重存储提供的数据 |
| `plan` | worker 获取 plan，以及 backend 与持久 batch 的必要调用 |

本轮收缩：删除共享 `ledger.py`（账本插件尚是本组内部实现），`common_prefix` 移到 `core.state.alloc.prefix`；删除 graph 的内部 `GraphDep`、强制公开的重录流程、重复的可选状态动作 Protocol，以及未实际交接的状态/模型预留字段。请求表与账本的内部字段不建立共享镜像。状态声明先保留 request/slot，跨层借用、host 放置和分区等在实际实现需要交接时再加入。

## 4. 每步记录（两边最重要的接缝）

代码：`core/contracts/records.py`（调度 ↔ worker）、`requests.py`（前端 ↔ 调度）、`sampling.py`、`columns.py`。

### 4.1 为什么要先定这个

vLLM 从 V0 重写到 V1，CPU 开销是主要原因之一。2024 年 9 月 vLLM 公布的 profile 里（Llama 3 8B、一张 H100），GPU 执行只占 38%，HTTP 占 33%，调度和准备输入占 29%。V1 的做法有两条：
- 相邻两步的状态大部分相同，就复用，只发变化的部分；
- 能挪出核心循环的事，都挪出去。

SGLang v0.4 把调度和执行重叠起来：GPU 在跑第 N 步时，CPU 已经在安排第 N+1 步。vLLM 的 Model Runner V2 设计文档也承认，V1 的异步调度是后来补上的，所以留下了复杂度。

我们比它们更需要这一条：调度宿主和 worker 是两个进程。不重叠的话，每一步的关键路径上都要多一次跨进程往返，再加上调度器的全部 CPU 时间。所以契约从一开始就按“最多提前一步”来设计，哪怕阶段 2 默认跑的是同步循环。

[step-overhead.md](../evaluations/step-overhead.md) 的结论（“保持同步”）只针对一件事：进程宿主的循环用 asyncio 还是同步。它测的是空 `step()` 外面的框架开销，不涉及 CPU 和 GPU 的重叠。同步的循环完全可以做重叠：mini-sglang 的 `overlap_loop()` 就是一个普通的 `while` 循环，靠异步 launch 和晚一步收结果。

### 4.2 记录和分段

每步记录是开放记录（[ipc.md](ipc.md) §3）。epoch 就是步号，由调度方分配。

| 方向 | schema | 分段 | 内容 |
|---|---|---|---|
| 调度 → worker | `sched.step_plan@1`（每 rank 一份） | `sched.rows@1`（`RowDelta`） | 持久 batch 的增量：新增的行（请求、采样参数）、删掉的行；`full` 表示整体重发 |
| | | `sched.batch@1`（`BatchPlan`） | 这一步每行算几个 token、从哪个位置开始、输入 token；`from_previous`；分桶 |
| | | `state.ops@1`（`StateOps`） | 每种状态新分给各行的条目（页、槽位），以及条目拷贝（前缀命中时恢复快照、块边界打快照） |
| worker → 调度 | `worker.step_result@1` | `worker.sampled@1`（`Sampled`） | 每行采到的 token（开投机时是接受的长度加一） |
| | | `worker.stats@1`（`StepStats`） | 这一步的 CPU 时间、GPU 时间，以及开始前 GPU 空了多久 |
| | | `state.capacity@1`（`Capacity`） | 各状态种类可用的条目数，启动时和 grow 以后发 |
| | | `worker.resync@1`（`Resync`） | 持久 batch 被重建了，要一份全量 |
| 前端 → 调度 | `frontend.requests@1` | `frontend.batch@1`（`Requests`） | 新请求（已分词）、中止 |
| 调度 → 前端 | `sched.tokens@1` | `sched.events@1`（`TokenEvents`） | 这一步各请求的新 token、结束原因，以及这一步跑在哪个身份下 |

四条都走每步记录流：有序、不丢、单生产者、背压。

`BatchPlan` 的 token 怎么按行拆（2026-11-08，PR #19 评审）：
- `from_previous = 1` 的行只算 1 个 token，不从 `tokens` 里取；
- 其余的行按行序，从 `tokens` 里依次各取 `num_tokens` 个；
- 所以 `tokens` 的长度正好是这些行的 `num_tokens` 之和。

`BatchPlan.check()` 检查这几条。worker 在测试和调试模式下对每份计划都查一遍。

投机的草稿 token 不在这一版里。草稿放在哪、验证步的 `num_tokens` 怎么数，到 M2d 做投机时再定，随新版本的分段一起出（§4.4 第 7 条）。

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
  - 采样不在 worker 里留随机状态：客户端没给 seed 的话，调度器在接纳请求时补一个，worker 按（seed, 位置）取随机数。所以重建前后采到的 token 一样，和 batch 怎么拼也无关。
  - 开投机时，下一步要验证的草稿如果留在显存里，也一起丢掉；重建后每行的第一步不带草稿。
  - 契约测试（M2a，`torch_cpu`）：请求 decode 若干步以后重建 batch，再接着生成，和不重建的对照逐 token 一致；要覆盖重建后隔了几步才再排进 batch 的行，以及开重叠的情况。

### 4.4 一步重叠

1. **深度**：调度器收到第 N−1 步的结果之后，才能发第 N+1 步。所以没收到结果的计划最多两份：一份在跑，一份在排队。
2. **future token**：`from_previous = 1` 的行，输入 token 就是 worker 上一次给这一行采到的 token，在显存里直接接上，不回 host。不开投机时，位置是确定的（加一），调度器自己算得出来。
3. **提前分配**：第 N+1 步要的页，调度器先按“请求还会继续”分出去。如果请求在第 N 步已经结束，这些页按第 5 条延迟释放。
4. **结束晚一步**：请求在第 N 步采到了停止 token，调度器要等第 N 步的结果才知道，这时第 N+1 步已经发出去了。多算出来的那个 token 由调度器丢掉，不发给前端（`TokenEvents` 的说明）。前端在反分词时发现停止字符串，发 `AbortRequest`，处理方式相同。
5. **延迟释放**：`AccountingPlugin.release(request, last_step)` 只做标记。等最后一个用到这些条目的步的结果到了（`retire(step)`），它们才回到空闲表。否则新请求拿到这个槽位，会被还在跑的旧步写坏。
6. **暂停**（PR #18）：
   - 调度器的 `quiesce()`：不再发新计划，把已发计划的结果都收完，返回最后一个做完的步号。
   - worker 的 `quiesce()`：等 GPU 上的工作都做完。
   - 领头的调度器先停，跟随的 worker 跑到同一步再停。所以执行阶段卸掉的东西，不会还有在途的工作在用。
7. **投机解码**：开投机时，下一步的位置取决于这一步接受了几个草稿 token，这个数只在 GPU 上才知道。v0 规定开投机时不重叠。契约给它留了位置：以后位置也可以写成引用，那时加一个新版本的分段。
8. **同步循环是特例**：收到第 N 步的结果才发第 N+1 步，记录完全相同。阶段 2 默认走同步循环；M2d 加开关 `sched.overlap`（live 键），并加一组对比：开重叠和不开重叠，token 完全一致（要求所用的 kernel 和 batch 怎么拼无关）。
9. **可观测**：`StepStats` 给出每步的 CPU 时间、GPU 时间、GPU 空闲时间。RSI agent 靠它判断瓶颈在不在 CPU，再决定开不开重叠、怎么调。

### 4.5 按列编码

- 每行一个值的字段都编码成列（`Column`）：一段带类型的字节串，用标准库的 `array.array` 构造，读的时候可以零拷贝（`torch.frombuffer`）。一批 256 个请求时，不要每步给每个请求建一个 Python 对象，那是 vLLM V0 走过的老路。
- 每步只有新增的请求（`AddedRow`、`NewRequest`）是逐个的对象，它们的数量很小。
- 小端字节序。

## 5. 状态

代码：`core/contracts/state.py`（声明）、`storage.py`（存储单元）；`Binding` 在 `tensor.py`。账本实现与辅助算法在 `core/state/alloc/`，不放进共享 contracts。

### 5.1 声明

每层声明自己要存的东西（`StateDecl`），存储单元和记账插件用同一份声明：
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
- 可选：`grow`、`refit`、`relocate`、`export_state` / `import_state`，用 `capabilities.actions` 和 `[state] migrate` 声明。
- **绑定代数**：交出去的 buffer 只要有一个换了地址、shape 或 dtype，代数就加一；同地址换内容不加。graph 在 replay 前，每个提供者只比对一个整数（§9.3）。
- **scratch 条目**：capture 前的 warm-up 会真的跑起来，GDN 的 recurrent state 会被推进，KV 会被写入。如果用在途请求的条目来 warm-up，它们的状态就坏了（探针 P6）。所以每种状态都留出 scratch 条目，永远不分给请求。

### 5.3 记账插件（调度宿主里，`ledger.<kind>@1`）

2026-11-08 更新：以下是 M2d 前缀缓存的设计目标。池与联合准入算法归 `core.state.alloc` 实现所有，不预先要求拆成独立单元；共享 `AccountingPlugin` 已删除。需要独立替换插件时，再围绕实际交接提取必要接口。本 PR 只保留已有的 `common_prefix` 辅助算法，账本实现由后续 PR 提供。

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
- 前缀索引（块哈希 → 条目）由各记账插件自己持有，因为每种状态对齐和存前缀的方式都不同。`core.state.alloc` 只负责编排。
- 条目分三种：
  - 空闲；
  - 被请求持有：可以同时被几个请求持有；预留的快照槽由它快照的那个请求持有，直到发布；
  - 已缓存：发布过，没有请求持有，可以淘汰。

  插件只在要分出条目时才淘汰，按最久未用。被持有的条目、待拷贝的源条目都不淘汰。
- **接纳**：先用 `common_prefix` 定下长度 L，再对每种状态调 `admit(request, block_hashes, L, new_tokens, step)`。
  - 它返回两样东西：按表顺序排好的条目（先是复用的，再是新分的），放进 `StateAssign`；这一步之前要做的拷贝，放进 `StateCopy(before)`。
  - 要么全给，要么返回 `None`、什么都不拿。只要有一种状态返回 `None`，这个请求就不接纳，已经接纳了它的那几种状态调 `cancel(request, step)`，拿到的东西当场退回。
  - 不能用 `release(request, step)` 撤销（2026-11-08，PR #19 第三轮评审）：`release` 要等这一步的结果，可这一步如果谁都没接纳，可能根本不会发出去，预留的槽和快照引用就一直放不掉。`cancel` 只用于还没发出去的那部分计划；规划时拿了条目、最后又没排进计划的行，也用它退回。
  - KV：复用的页原地读，请求一直持有到 `release`。
  - GDN：新分一个槽，拷贝源是缓存的快照。快照被持有到这一步的结果回来，所以拷贝跑完之前不会被淘汰。
- **快照和发布分两步**（2026-11-08，PR #19 第三轮评审）。开了重叠时，第 N 步的结果还没回来，调度器就要规划第 N+1 步，而 N+1 的输入 token 还在 worker 上（`from_previous`），所以规划时算不出覆盖它的块哈希。只有拷贝必须在规划时定，哈希可以晚一步：
  - **快照**（规划时）：每行这一步的 chunk 末尾在 `upto`，调 `snapshot(request, upto, step)`。不需要哈希。
    - GDN：`upto` 正好是快照间隔的倍数时才打快照（§5.4）。插件预留一个空闲或可淘汰的槽当目标，返回 `StateCopy(after)`；预留不到就不打。
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
- **契约测试**（M2d，两个记账插件）：
  - 请求 A 算完一段前缀、发布、结束；请求 B 命中这段前缀，拿到的是 A 的页和快照槽；B 的拷贝和使用没做完之前，这些条目不会被回收或淘汰。
  - 联合准入里第一种状态接纳了、第二种拒绝，这一步不发：第一种的容量和缓存引用当场恢复，随后一个小一点的请求仍能接纳。
  - 开重叠，推迟前一步的 `Sampled`，让下一步正好落在快照边界：token 未知时不登记哈希；最后发布的位置、哈希和状态，和同步循环逐项一致。

### 5.4 Qwen3.8 的状态

- `kv.paged`：16 层 full attention，每层的 K、V。24 个 q 头、4 个 kv 头，head dim 256。
- `gdn.slots`：48 层 GDN，每层的 conv state `[槽, 10240, 3]` bf16 和 recurrent state `[槽, 48, 128, 128]` fp32。TP1 时每个请求约 147 MiB，256 个并发就是约 37 GiB（探针报告第 9 条）。所以槽位数在启动时由 `core.state.holder` 按显存定，以后可以像 KV 一样用 VMM 扩槽位。
- 前缀复用：GDN 状态只在块边界有快照。快照占同一个池里的槽位，打快照、恢复快照都是条目拷贝（`StateCopy`）。所以 `gdn.slots` 的 `reusable` 只能对齐到快照边界。
- **快照只能在一步的末尾打**（2026-11-08，PR #19 评审）：
  - GDN 的状态在一步之内原地往前推，`StateCopy(after)` 只拷得到这一行在这一步 chunk 末尾的状态。
  - 例：一个 256 token 的 chunk 跨过了 64、128、192。步末时这几处的状态已经没有了，不能把步末的状态当成它们存进缓存。
  - 规则：chunk 的末尾正好落在快照边界上才打快照，chunk 里面跨过的边界不打。
  - 想在哪里留快照，就由分块决策点把 chunk 切在那里。默认 chunk 长度取快照间隔的倍数，并把 prompt 里最后一个快照边界切成一个 chunk 的末尾。这样，共用同一段 system prompt 的后来请求能从最靠后的边界恢复。
  - decode 每步只算一个 token，正好落在边界上的那一步打快照。开投机时，一步接受几个 token 事先不知道，所以 v0 开投机时 decode 期间不打快照。
- MTP：config 里有 1 层 MTP，它的 KV 是 `kv.paged` 的另一层声明。

### 5.5 显存里怎么排（@gaow0007 定）

- 包括：页大小、NHD 还是 HND、GDN 状态在槽内的排布、VMM 的映射粒度（2 MiB）和每次扩多少。
- 都写成声明里的数据，不写死在代码里。
- 存储单元由 @wutianyuan1 这边实现，布局由 @gaow0007 定。

## 6. kernel 目录

代码：`core/contracts/kernels.py`。schema 和 flashinfer-bench 的 Definition / Solution 对齐。

- **definition** 写算什么：名字带版本、语义说明、PyTorch 参照实现（必须有）、容差、结果和 batch 怎么拼是否无关。参照实现在 `torch_cpu` 上跑，是 CI 的基准，也是检验 solution 的标准。
- **solution** 写怎么算：用哪个库的哪个版本、支持哪些 shape、能不能录进 graph、需要哪些平台能力、要不要进程级初始化、能不能跨机。
- 选择：编 plan 时按 definition 加 shape 选定，eager 路径每次调用都查。选不到合适的就退回参照实现，并在 plan 和结果里标出来。选中的 solution 是 plan 的按值依赖。
- kernel 单元通过 `ctx.register` 登记到目录里，层只按 definition 调用，不直接调具体的 kernel。

## 7. 层、模型、权重

代码：`core/contracts/model.py`。

- 逻辑张量 = 格式（插件）+ 放置（数据）。层不持有权重，只从权重存储拿视图，所以 refit 是原地换内容，graph 不用重录。
- 模型单元在启动时给出 `Assembly`：用哪些层、每层存什么状态、有哪些权重。各项尺寸从 checkpoint 的 config 读，不写死。
- Qwen3.8：64 层，48 层 GDN、16 层 full attention，排列是 3 层 GDN 加 1 层 full attention 重复。M2a 用同样的结构、小尺寸、随机权重。

## 8. attention 和 GDN 的 backend（@gaow0007 审）

代码：`core/contracts/plan.py` 的 `Backend`。

- 照 sglang 的分工：
  - **graph 外**：每步在 host 上准备。从持久 batch 和 `BatchPlan` 算出 page 表、序列长度、槽号，写进本单元自己固定地址的 buffer（`prepare`）。
  - **graph 内**：只做形状固定的 GPU 操作（`forward`）。
- 这些固定 buffer 是 graph 的按地址依赖，由 backend 的 `Binding` 交出。batch 大小是按值依赖。
- flashinfer 在 cuda-graph 模式下重新 `plan` 以后可以直接 replay，同一 batch 大小内不同序列长度都正确（探针 P5、P6），所以 flashinfer 的 plan 结果不是按值依赖。
- 有了持久 batch 以后，page 表可以在显存里增量维护，`prepare` 只更新变了的行。

## 9. plan 单元和 CUDA graph（@gaow0007 审）

交接见 `core/contracts/plan.py`。graph 的依赖表、重录迭代器和缓存组织留在 plan 实现内部。

### 9.1 plan

- 每个（阶段, 分桶）一个 plan。
- decode 录 graph，prefill 走 eager。
- 分桶由决策点“graph 分桶”给出，默认取 2 的幂。
- plan 是派生产物，依赖登记给 `core.deps`。

### 9.2 录 graph

- capture 用的 stream 由平台提供（§10.4）。warm-up 和 capture 用同一个 stream，cuBLAS workspace 就不会随重载增长（探针 P3）。
- warm-up 只用 scratch 条目（§5.2）。
- graph 的输出算作 graph 的一部分：释放 graph 时连输出一起放掉，否则它的私有显存池放不掉（P3）。
- 新录的 graph **不和仍可能 replay 的旧 graph 共享显存池**。共享的话，旧 graph 不按录制顺序 replay 时，会覆盖新 graph 的输出（P3）。
- 平台的 capture 原语**不在每次录制前清显存缓存**。`torch.cuda.graph` 每次进入都会调一次 `empty_cache()`，P9 里有一次花了 5.7 s。可以改用 `CUDAGraph.capture_begin` / `capture_end`，自己决定什么时候清（还没测过）。
- **capture 时，同一个进程里不能有别的 CUDA 工作**（P10）。默认的 `global` 模式只检查 CUDA 标记为 capture-unsafe 的 API 调用，不能当作保护。所以不在后台线程里录，只在 §9.4 的暂停窗口里录。

### 9.3 replay 前检查

- 依赖分两种：
  - 按绑定：每个提供者（状态单元、backend、持久 batch、权重存储）一个代数；
  - 按值：选中的 solution、单元实例（也就是代码版本）、录 graph 时读到的配置键、batch 大小，以及通信器（多 rank 时）。
- 通信器是资源依赖图里的一类派生产物；拆除时先拆用它的 graph，再拆通信器。
- replay 前只比对这些，目标是每步几微秒。探针 P6 逐个 tensor 比对 55 项，每步要 38 µs；真模型的依赖会有几百项。
- 逐个 tensor 比对只在测试和调试模式下打开。
- 必须检查：真 graph 读到已释放或已换地址的显存时不报错，只会读到垃圾（P1）。模块卸掉以后，旧 graph 照样跑旧 kernel（P5）。

### 9.4 重录

- **只在步与步之间的暂停窗口里录，一个窗口录一个分桶**：
  - `rebuild` 返回一个迭代器，每个安全点前进一步（[change.md](change.md) 4.6）；
  - 每一步先等在途的那一步在 GPU 上做完（开了重叠的话，这个窗口里重叠暂时中断），再 warm-up、再录；
  - 旧 graph 在窗口之间照常服务，到 epoch 边界再换成新的。
- **哪些变更能分片重录**（2026-11-08，PR #19 评审）：
  - 分片的前提是：这次变更不重载 `core.plan`，也不重载它用到的单元。参与者会把被重载单元的准备动作推迟到执行阶段，在新实例上一次做完（`Participant._prepare_or_defer`，报 `change.prepare_one_shot`）。
  - 能分片：
    - 新 kernel 做成一个新单元，再加一个选它的配置项（场景 ④，`tests/scenarios/test_scenarios.py::test_4_a_new_kernel_rebuilds_only_the_graph_that_uses_it`）；
    - 只改分桶的配置；
    - 在目录里已有的 solution 之间换。

    新的值由重录动作的参数带进来。
  - 不能分片：原地改了 `core.plan`、backend、模型、层、`core.kernels` 或 `worker.batch` 的代码，或者改了它们 import 到的 `core.contracts` 模块（import 得到这个模块的单元一起重载，§3.1）。这时重录在暂停期内一次做完，P9 的尺寸下 36 个分桶要 6.9 s 以上。
  - 不能分片时，v0 有两种做法，都不用改框架，M2c 由 @gaow0007 选定：
    1. 暂停期内只录最常用的几个分桶（按分桶的命中次数挑），其余分桶恢复后先走 eager。worker 循环在步与步之间调 `core.plan`，一个窗口补录一个分桶，每个窗口的代价和准备阶段分片一样，约 0.2 s。补录完之前，这些分桶的步慢一些，但结果不变，因为 graph 和 eager 逐位一致。补录不是变更：graph 是 eager 的缓存。
    2. 整段停下来全录完，dry-run 把停顿报出来。
  - 同一个单元的新旧实例并存（旧实例继续服务，新实例在旁边准备）要改框架，v0 不做。RSI agent 想让改代码也能分片准备，就照场景 ④ 出一个新单元，旧单元以后再删。
- **代价**（P9，Qwen3.8-27B 的尺寸，sglang 默认的 36 个 batch 大小）：
  - 平时一步 30.6 ms。每个窗口约 0.2 s：等在途的那一步约 30 ms，两次 warm-up 约 110 ms，录制约 47 ms。
  - 36 个窗口合计约 6.6 s；一次全部录完也要 6.9 s 以上。
  - 所以带重录的变更，要把“几十个约 0.2 s 的停顿”或者“一次几秒的停顿”算进对在途请求的影响，并在 dry-run 里报出来。这条规则加进 `core.deps`。
  - 每个版本的 warm-up 可以考虑只做一次，不必每个分桶都做两遍（还没测过）。
- **切换**就是一次普通的暂停：领头者先停，循环单元 `quiesce()`（PR #18），然后执行、换上新 graph。P9 切换时 GPU 恰好已经同步过，所以“GPU 正忙时停下、切换”还没在 GPU 上验证过，放进 M2c 的 GPU 测试。
- **显存峰值**：准备期间是新旧两套显存池之和；改了通信器的话，还要加上新旧两套通信器，每个 rank 每个通信器约 0.4–1 GiB。放不下就推迟到排空以后（[change.md](change.md) 4.7）。
- 另一种做法是在别的进程里录，但那样要跨进程共享状态的地址，v0 不做。

## 10. 平台 v2（@gaow0007 审）

### 10.1 三个平台

| 平台 | 用途 | 有没有计算 | graph |
|---|---|---|---|
| `fake`（已有） | 测引擎的机制：变更、重载、故障 | 没有，只有按名字调用的假 kernel | 记录依赖，replay 时检查 |
| `torch_cpu`（新，@wutianyuan1） | 在 CPU 上跑真模型语义：M2a、CI 里的单元测试和五组对比 | torch CPU 张量、参照 kernel | 记下 capture 时的调用，replay 时检查依赖、重新执行 |
| `cuda_sm100`（新，@gaow0007） | SM100 | torch.cuda、CUDA VMM | 真 CUDA graph |

- `torch_cpu` 和 `cuda_sm100` 跑同一套平台契约测试。
- `torch_cpu` 上，“graph 对 eager”自然完全一致，这组对比只在 GPU 层有意义。

### 10.2 buffer

- 在 torch 平台上，`Buffer` 背后就是一个 torch 张量，平台通过 `tensor` 把它交出来（类型不透明）。
- 假平台照旧，没有张量。

### 10.3 VMM 和 grow

- `VirtualRange` 给两种 view：
  - `view`：只覆盖已映射的部分，什么操作都能用；
  - `span`：覆盖整个预留区间，只给按页号访问的 kernel 用。
- 在 GPU 上，eager 操作只要碰到没映射的部分（例如对 span 做 `zero_()`、`sum()`），就会触发页错误，CUDA context 直接废掉，进程就丢了。
- 所以 `fake` 和 `torch_cpu` 要模拟这一点：kernel 访问超出已映射部分就报错，CI 才能抓到。
- 现在假平台不允许建超出已映射部分的 view（`engine/platforms/fake/__init__.py:82`），要按上面改。
- grow 不改地址，也不改绑定代数；grow 之前录的 graph 能读到新映射的页，不用重录，不用排空（P4、P6）。
- 只有 span 能看到新页：graph 里写死了 view 的 shape。
- 持有方放掉引用以后，要等最后一个 view 也释放了，才解除映射（P4）。

### 10.4 stream

- 平台为每张卡提供数量有限、可以复用的 stream，按用途分开：执行、拷贝、capture。这样既不限制 CPU 和 GPU 的重叠，stream 的数量也是固定的。
- 单元不许自己建 stream。`core.manifest` 新加一条规则：单元的代码里出现 `torch.cuda.Stream(` 就报错。
- 原因：每个跑过 cuBLAS 的 stream 都各缓存一份 workspace，重载会让显存一直涨，最多涨约 270 MB（P3）。stream 的数量固定下来，workspace 也就不会涨。
- 现有的 `Platform.stream()` 改成按用途交出平台的 stream。

### 10.5 graph

- `graph().capture()` 每一代 plan 用自己的显存池。replay 前按 §9.3 检查。
- `fake` 也改成按绑定代数记依赖；同时保留按地址的检查，用来抓“某个提供者忘了加代数”。

### 10.6 收尾和清理

- `platform.synchronize(device)`：给 worker 的 `quiesce()` 和 §9.4 的暂停窗口用。
- **卸载单元时不清 cuBLAS workspace。** `_cuda_clearCublasWorkspaces()` 清的是整个进程的全局缓存，还活着的 graph 和在途的工作可能正在用其中的地址（P3）。只有确认没有任何 graph 或工作会再用到时才能清，例如进程退出时。

### 10.7 撤不掉的注册

- `torch.library` 的注册不会随模块消失，不撤销的话，重新 import 会报“重复注册”（P2）。
- 这类注册必须通过 `ctx.effect` 登记撤销函数（[unit-runtime.md](unit-runtime.md) §5），并加一条植入缺陷的测试。

### 10.8 代码重载

- 约束：快照路径下的内容写好以后永不改变。
- 原地改写同一个 `.py`，同一秒内、长度不变时，会 import 到旧的字节码（P7）。
- 现在的做法满足这条约束：每个身份一个快照目录；节点代理设了 `PYTHONDONTWRITEBYTECODE=1`；change-set 和 blob 都排除 `__pycache__`。控制面进程会往自己的快照目录写 `.pyc`，这不要紧，因为那里的源码不会再变。
- 物化本身就保证了这一点：节点代理和控制面都按摘要给快照目录命名，先写临时目录再原子改名，已经存在就不重写；节点代理还把目录设成只读。测试 `tests/framework/cluster/test_cluster.py::test_a_snapshot_is_written_once_and_never_rewritten` 把它固定下来：同一身份再物化一次，文件不被改写，目录是只读的。

## 11. 测试和工具链

### 11.1 测试分层

| 层 | 位置 | 平台 | 谁跑 |
|---|---|---|---|
| 单元的 CPU 测试 | `tests/units/<单元路径>/` | `torch_cpu`、`fake` | CI |
| 引擎级 | `tests/scenarios/` | `fake`，M2a 以后加 `torch_cpu` | CI |
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
| adapter 权重放在显存的固定槽位里，请求共用 | 一种 `slot` 寿命的状态。记账插件按请求数给槽位记引用计数；“请求要的 adapter 已在槽位里”和 KV 一起参与联合准入 | 装得下 |
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
| 5. grow 的契约，两种 view 分开给，假平台跟着改 | §10.3 |
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

1. §9.4 里还没测过的几件事：capture 前不清显存缓存、每个版本只 warm-up 一次、GPU 正忙时停下并切换；dry-run 怎么报重录造成的停顿；不能分片时选哪种做法。
2. `torch_cpu` 平台上的 graph：要不要模拟“录制时写死 Python 标量”。不模拟的话，这类错误只有 GPU 层才抓得到。
3. 前端和调度之间的流在 M2a 用每步记录流。前端以后要不要拆成多个进程，到时再定。
4. 采样放在 graph 里还是 graph 外，由 @gaow0007 在 M2c 时定；契约只要求采到的 token 留在显存里，给 `from_previous` 用。
5. 投机的记录：草稿放在哪、验证步的 `num_tokens` 怎么数。M2d 定，出新版本的分段（§4.2）。
