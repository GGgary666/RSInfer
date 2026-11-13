# 前缀复用：照 SGLang 搬（计划，2026-11-13）

- 状态：实现了（2026-11-13，#59），CPU 和 4090 D 上验收过，SM100 还没有。属于 M2d（[stage2-contract.md](stage2-contract.md) §1.3）。
- 来源：SGLang `e838360`，和 kernel 是同一个版本（`~/sglang-e838360/python/sglang/srt/`，下面的路径都相对于它）。Qwen3.5 结构在开重叠或 `page_size > 1` 时，SGLang 默认用的那一套：radix cache 加 mamba 状态的 `extra_buffer` 策略。
- 取代：stage2-contract.md §5.3 前缀缓存的设计，也就是块哈希、`reusable` / `common_prefix`、快照与发布；还有 §5.4 的"快照只能在一步的末尾打"。和 §5.5 一样，改成严格照 SGLang。
- 继续有效：§5.3 末尾的账本验收测试（按树的说法改写）、§5.5.4 的规则（共享页不续写、恢复到请求自己的可写状态）、§5.5.6 的 M2d 部分，以及五组对比里的"前缀命中对不命中"。

## 1. SGLang 怎么做

### 1.1 用哪种策略

- **哪一套**：
  - `Qwen3_5ForCausalLM` 在 `_MAMBA_EXTRA_BUFFER_ARCHS` 里（`arg_groups/overrides.py`）。
  - `mamba_radix_cache_strategy = auto` 时，开重叠或 `page_size > 1` 就选 `extra_buffer`；否则选 `no_buffer`，那个只支持 page 1、而且要关重叠（`_mamba_radix_cache_resolution`、`arg_groups/mamba_hook.py`）。
  - `extra_buffer` 要求 GDN 用 Triton（`supports_mamba_cache_extra_buffer`）。
- **我们这边**：page 16，可以开重叠，GDN 是 Triton。所以对应 `extra_buffer`，只搬这一种。
- **默认值**：radix cache 默认开（`disable_radix_cache = False`）。调度策略默认 `fcfs`，不因前缀缓存而改。

### 1.2 树

代码在 `mem_cache/unified_radix_cache.py`、`unified_cache/unified_tree_core.py`、`unified_cache/components/{full,mamba}.py`。

- **结构**：按 token 的 radix 树，按页切。节点上挂两样东西：
  - Full：这一段的 KV 页号；
  - Mamba：一个 GDN 槽号，可以没有。
- **只缓存整页**：请求最后不满一页的部分留在请求自己那里（`radix_cache.py` 的 `page_aligned`）。
- **匹配**：
  - KV 命中的长度是沿树走到的最长整页前缀。
  - 可用的前缀是这条路径上最深的、挂着 Mamba 状态的节点（`MambaComponent.create_match_validator`）。所以前缀长度由 GDN 状态决定，KV 多命中的那部分不用。
  - 最多复用到 prompt 长度减一（`_compute_max_prefix_len`），至少留一个 token 来算 logits。
- **分叉点**（`mamba_branching_seqlen`，`finalize_match_result_in_tree_core`）：
  - KV 命中比可用前缀长时，取 KV 命中里最后一个检查点网格上的位置。
  - 这个请求 prefill 时在那里多存一份状态。之后走同一条路的请求就能用到更长的前缀。
- **加锁**：请求用着的节点加 `lock_ref`，锁住的不淘汰。
- **淘汰**：
  - 只在分不到页或槽时淘汰（`evict_for_alloc`），按最久未用。
  - KV 只淘汰叶子。Mamba 状态可以单独从中间节点拿掉，节点保留 KV。
  - `mamba_max_states_per_path`：一条路径上最多保留几份状态，默认 -1，不限。

### 1.3 一个请求占几个 GDN 槽

- **槽数**：一个运行槽（`mamba_pool_idx`），加上 ping-pong 的跟踪槽：开重叠 2 个，不开 1 个（`allocation_sizing.get_mamba_tracking_slots`）。
- **最大运行请求数**：不超过 `max_mamba_cache_size // 比例`（`kv_cache_configurator._calculate_mamba_ratio`）。
  - 开前缀缓存时，比例是 3 加跟踪槽数：开重叠 5，不开 4。
  - 关掉时是 1。

### 1.4 什么时候存状态（跟踪）

- **检查点网格**：`lcm(max(64, page_size), page_size)`（`runtime_context.mamba_cache_chunk_size`、`mamba_checkpoint_grid`）。page 16 时是 64。
- **prefill**（`ScheduleBatch._mamba_radix_cache_v2_req_prepare_for_extend`）：
  - 位置：这一步 chunk 里最后一个网格位置，即 `prefix_len + extend_len // 网格 * 网格`。分叉点落在这个 chunk 里时，存分叉点。
  - SSM 状态：位置正好是 chunk 末尾时拿最终状态，否则从 chunk kernel 的中间状态 `h` 里取。
  - conv 状态：从这一步的输入里取那个位置之前的 conv 窗口。
  - 实现在 `hybrid_linear_attn_backend.py` 的 `_init_track_ssm_indices`、`_init_track_conv_indices`、`_track_mamba_state_extend`。
  - 所以不需要按网格切 chunk，chunk 怎么切都能存。chunk 比网格短时 SGLang 打 warning。
- **decode**：
  - 位置每到 `mamba_track_interval`（默认 256，和网格取最小公倍数）的倍数，这一步结束时把运行槽拷进跟踪槽。一个 kernel 拷全部层（`track_mamba_states_all_layers`，在 `kernels/ops/mamba/mamba_state_scatter_triton.py`）。
  - 开重叠时，下一步用另一个 ping-pong 槽。

### 1.5 什么时候进树、怎么用

- **进树**（`mem_cache/common.py`）：
  - prefill 的一个 chunk 结果回来、请求还没结束时：`checkpoint_kv_cache`。
  - 请求结束时：`release_kv_cache` → `on_release`。
  - decode 期间不进树。
- **进树的内容**：
  - KV 按整页插进树。
  - 跟踪槽里的状态"捐"给树上对应长度的节点，请求那个 ping-pong 位置换一个新槽（`donate_tracking_slot`）。
  - 结束时，从两个跟踪槽里挑一个不超过已缓存长度的（`_select_finished_checkpoint`）。
- **命中以后**：
  - GDN：请求新分一个运行槽，在 forward 之前、同一条 stream 上把树里的状态拷进去（`mamba_cow_src_index`、`finalize_match_result_in_cache`）。
  - KV：页直接共享，只读。前缀是整页对齐的，请求从下一页开始写。

## 2. 我们这边怎么对上

分工不变：账本在调度宿主里，只管编号；worker 的存储单元管显存。SGLang 的树存的也只是编号，这一点一样。

| SGLang | 我们 | 要做的 |
|---|---|---|
| 树（Full + Mamba 两个组件，只要设备上的部分） | `core.state.alloc` | 树、匹配、插入、加锁、淘汰都放进账本；现在的 `Pool`（mini-sglang 的空闲表）照旧管空闲编号 |
| `Req.kv` 的 `prefix_len` 和 mamba 字段 | 账本里每个请求的 `Holding` | 加跟踪槽、上次跟踪的位置、分叉点 |
| `ScheduleBatch` 准备 extend / decode 时算的跟踪位置 | 账本的 `reserve` | 跟踪位置由账本定，调度器只转交 |
| `mamba_cow_src_index`（forward 前拷） | 已有的 `StateCopy(before)` | 不改。`executors.pytorch` 已经照它恢复，并且不清这种槽 |
| prefill 和 decode 的跟踪 | 新的 `StateOps.track` | contracts 加字段，见 §2.1 |
| `checkpoint_kv_cache`、`on_release` | 结果回来、`retire` 之后的 `commit`；`release` | 账本新接口 |
| `_calculate_mamba_ratio` | 账本的 `max_requests`（实现时从 holder 挪到了账本，见 §7） | 开前缀缓存时，按同样的比例给出最多同时跑几个请求 |
| `--disable-radix-cache`、`--mamba-track-interval`、`--mamba-max-states-per-path` | `alloc.radix_cache`、`alloc.mamba_track_interval`、`alloc.mamba_max_states_per_path` | startup 键，改了要排空、重载（普通变更） |

### 2.1 接口（`core.contracts`）

`core/contracts/ledger.py`：
- `admit(request, tokens, step)` 改成拿 token 本身：
  - 先匹配，再接纳；全有或全无不变。
  - 返回前缀长度、每种状态的条目（复用的 KV 页在前），以及这一步之前要做的拷贝（GDN 恢复）。
- `reserve(request, upto, step)` 再返回这一步的跟踪：哪个跟踪槽、存到哪个位置；不跟踪就没有。
- 新加 `commit(request, tokens, upto)`：一步的结果回来以后，把已经算完的整页和跟踪到的状态插进树（`checkpoint_kv_cache`）。
- `release` 插最后一次（`on_release`）。
- 回了 `Resync` 的步什么都没写：不插，它的跟踪槽作废。

`core/contracts/records.py`：
- `StateOps` 加 `track: list[StateTrack] = []`。
- `StateTrack` 的字段：`kind`、`rows`、`dst`（跟踪槽）、`upto`（每行要存的位置，绝对位置，不含）。worker 在这一步里把 `upto` 处的状态写进 `dst`。
- 从 `h` 取还是拿最终状态、SGLang 那个 `+1` 的下标算法（`_force_track_h`），都是 worker 内部的事，不进接口。
- 兼容：新字段有默认值。变更只作用于空引擎（[empty-apply.md](empty-apply.md)），账本和 worker 在同一次变更里一起重载，不会有新账本配旧 worker 的时候。

### 2.2 worker

- **`layers.gdn`**：照 §1.4 搬 SGLang 的跟踪。
  - prefill：从 `h` 和最终状态取 SSM 状态，从输入取 conv 窗口。我们的 `chunk_gated_delta_rule` 本来就返回 `h`（`kernels/triton/fla/chunk.py` 返回 `o, None, h`），`layers/gdn/unit.py` 现在把它丢掉了。
  - decode：搬 `track_mamba_states_all_layers` 进 `kernels.triton`，在最后一个 GDN 层之后发一次。
- **参照实现**（CPU，进 CI）：`kernels.reference` 的 chunk 版（fla 的 naive 实现）也要给出中间状态。
- **decode graph**：SGLang 在 graph 里做跟踪（mask 和目标放在固定 buffer 里）。
  - 先做到能跑：回放之后在图外发那一个拷贝 kernel，只在有行到了间隔的步才发。
  - 合进 master 之前搬进 graph，和 SGLang 一样（2026-11-13 用户定）。
- **`worker.batch`**：不变。
  - 跟踪槽不进持久 batch 的表，每一步由计划给出。
  - 几行共享同一页，现在的检查不拒绝。
- **`kernels.flashinfer`**：master 上没有 FlashInfer 的 GDN prefill。SGLang 的 `extra_buffer` 也只支持 Triton，不受影响。

### 2.3 调度器（`core.sched`）

- **接纳**：账本给出前缀长度，请求从那里开始算；第一个 chunk 的位置就是前缀长度。
- **每一步**：把账本给的跟踪放进计划的 `StateOps.track`。
- **结果回来**：`retire` 以后对这一步跑完的 prefill chunk 调 `commit`。请求结束时由 `release` 插树。
- **淘汰**：账本在分不到条目时自己淘汰，调度器不用管。

### 2.4 显存怎么分（实现在账本里，见 §7）

- **怎么算**：开前缀缓存时，照 SGLang 用 GDN 槽数除以比例（开重叠 5，不开 4）当最大行数的上限，启动时报出来（SGLang 打 warning）。
- **代价**：TP1 一个 GDN 槽约 147 MiB（stage2-contract.md §5.4）。SM100 上那次对比，状态池 92.44 GiB 分到 305 个槽（[real-kernels-sm100.md](../evaluations/real-kernels-sm100.md)）；开前缀缓存加重叠，最大行数就是 61，比那次用的 64 少。要 64 行就得多给 GDN 槽（`holder.state_ratio` 或 `holder.max_slots`），KV 页相应变少。

## 3. 不搬的

- **不在 M2d 范围里的**：
  - HiCache（主机内存和存储分层）、SWA、会话、PD 分离、DCP、PP；
  - int8 检查点、只放主机的检查点、lazy extra buffer、撤回时的备份；
  - `no_buffer` 策略（要 page 1、关重叠）。
- **投机**：M2d 另做。SGLang 要求 `mamba_track_interval >= speculative_num_draft_tokens`，到时候一起接。

## 4. 默认开不开

默认开，和 SGLang 一致（2026-11-13 用户定）。`alloc.radix_cache = false` 关掉。

GDN 槽照 SGLang 的比例算（§2.4），用户也同意了（2026-11-13）：开前缀缓存时，最大行数会被 GDN 槽数封顶。

## 5. 验收

- **账本**（CPU，进 CI）：
  - stage2-contract.md §5.3 末尾三条，改成树的说法；
  - 只缓存整页；
  - 可用前缀由 GDN 状态定；
  - 分叉点会存下来；
  - 淘汰不碰锁住的节点，Mamba 状态被拿掉以后节点还留着 KV；
  - `Resync` 的步不插树。
- **worker**（CPU 参照，进 CI）：
  - 跟踪写进去的状态，和单独算到那个位置的状态相同。覆盖 chunk 中间、chunk 末尾、decode 间隔三种位置。
  - 从树上恢复以后接着生成，和不命中时逐 token 相同（fp32）。
- **整个引擎**（像 `tests/scenarios/test_m2a.py` 那样跑在 CPU 上）：
  - 两个共享长前缀的请求，第二个命中（看它 prefill 的 token 数）；
  - 文本和关掉前缀缓存时相同，开不开重叠都测。这就是五组对比里的"前缀命中对不命中"。
- **GPU**（4090 D 上 4 层，SM100 上 64 层）：
  - 同样的对比，bf16 按 stage2-contract.md §1.3 的口径，加上 decode graph；
  - 命中以后首 token 的时间；
  - 和 SGLang 同配置比：`tests/gpu/bench_serving.py` 加一个共享 system prompt 的负载。
- **A/B/C**（stage2-contract.md §5.5.6）：C 植入三种缺陷：共享页被写、锁着的节点被淘汰、跟踪位置错一格。

## 6. 分几步做

1. **账本**：树、跟踪槽、新接口。调度器先不用，只跑 CPU 测试。
2. **worker**：跟踪（参照实现和 Triton 都做），测试对照单独算的状态。
3. **调度器接上**：`alloc.radix_cache` 默认开，整个引擎的对比测试进 CI。
4. **GPU 验收和性能**：decode 的跟踪搬进 graph。

四步都在一个 PR 里（2026-11-13 用户定），叠在这份计划的 PR（#58）上：实现的 PR 先合进 #58，#58 再合进 master。先在跑着的引擎上热改开发：改的都是单元，不碰框架本体（`engine/framework.toml` 里没有这些模块）。

## 7. 实现（2026-11-13，#59）

### 7.1 东西在哪

- **接口**：
  - `core/contracts/ledger.py`：`match`、`admit(request, prompt, tokens, step)` 返回 `Admitted`（前缀长度和条目）、`reserve` 返回 `Reserved`（新条目、恢复、跟踪），以及 `commit`、`flush`、`release(..., tokens)`、`align`、`max_requests`。
  - `records.py`：`StateTrack` 和 `StateOps.track`。
  - `plan.py`：`PersistentBatch.track`。
- **账本**（`core/state/alloc/radix.py`，`RadixTree` 和 `RadixLedger`）：
  - `alloc.radix_cache` 默认开，`false` 时是原来 `pool.py` 不带缓存的账本。
  - 另外两个键：`alloc.mamba_track_interval`（256）和 `alloc.mamba_max_states_per_path`（-1）。
  - 原来按 §5.3 设计写的 `prefix.py`（`common_prefix`）和它的测试删了。
- **调度器**（`core.sched`）：
  - 等着的请求按缓存命中的长度算起点；接纳时从前缀开始，`request.computed` 就是前缀。
  - 没写完提示词的 chunk 结束在页边界上。
  - 恢复和跟踪放进计划的 `StateOps`。
  - 一步退役后对算了提示词的行调 `commit`；结束时把算过的 token 交给 `release`。
- **worker**：
  - `worker.batch` 记下这一步的 `StateTrack`；`state.gdn_slots` 给出全部层的池子（`pools()`）。
  - `layers.gdn`：prefill 从 `h` 和输入里的 conv 窗口取；decode 在最后一个 GDN 层后发一个 kernel，eager 和 graph 都是，graph 的 mask 和目标是桶的固定 buffer。
  - kernel 目录新定义 `gdn.track_states@1`：
    - 参照实现不和 host 来回，能录进 graph：每个 lane 都写，没被选中的 lane 写到 padding 槽 0。
    - Triton 版是 SGLang 的 `track_mamba_states_all_layers`，原样搬过来。
  - `gdn.chunk_delta_rule@1` 的参照实现也给出 `h`，和 fla 一样。
- **部署**：`deploy/agxinfer.py serve --disable-radix-cache`。

### 7.2 和 SGLang 不一样的地方

- **时机**：位置在一步退役时才往前走。所以回了 `Resync` 的步能撤回：它的跟踪不算数，恢复的拷贝下一步再发。SGLang 没有 `Resync`。
- **跟踪槽**：总是两个，比例总是 5。`sched.overlap` 是 live 键，SGLang 在启动时按开不开重叠定。
- **重复的页**：树里已经有的 token，请求自己的页不交给树，到请求结束才放。worker 的表只往后加，SGLang 是把树里的页写进请求那一行。
- **prefill 末尾的 conv 状态**：从这一行自己的槽里拷，SGLang 从输入里取窗口。行比窗口长时两者相同，比窗口短时只有前者对。
- **最多几个请求**：账本到了数就不再接纳；SGLang 是启动时把 `max_running_requests` 限住。超过 `sched.max_batch` 时调度器在日志里说。
- **丢掉缓存**（`flush`）：每次提交的变更（排空的 `commit`，`refit` 也算）和换了 worker 时都丢，比 SGLang 只在换权重时丢更保守。空闲时 worker 每秒重报一次 `Capacity`，数一样就什么也不动。
- **淘汰**：没搬 `_cascade_evict`。中间节点只摘掉状态，留下 KV。
- **匹配**：每一轮规划都给等着的请求算一次命中，有空行时才算，同一版树只算一次。

### 7.3 测试（CPU，进 CI）

- **账本**：`tests/units/scheduler/test_radix.py`，9 条：命中前缀、分叉点、结束时的 decode 检查点、`Resync`、重叠时两个跟踪槽、淘汰不碰在用的前缀、5 个槽一个请求、坏配置、`flush`。
- **worker**：`tests/units/executors/test_prefix_track.py`。
  - 存下的状态和单独算到那里的相同，chunk 中间、chunk 末尾、decode 三种位置都测；
  - 从存下的状态恢复、共享页再算 4 个 token：状态和 logits 是整个提示词的，不恢复的对照差得远；
  - 不在 64 倍数上的位置被拒。
- **graph 路径**：`tests/units/executors/test_cuda_graph.py::test_a_captured_step_keeps_states_for_the_prefix_cache_as_the_eager_one`，用替身录制器在 CPU 上回放。它还查一个桶上次回放存过状态、下次不存的情况：清 mask 那一步拿掉，这条就失败。
- **端到端**：`tests/units/worker/test_prefix_cache.py`，真调度器加真 worker。
  - 第一个请求结束、空闲 2.5 秒以后，另外三个请求从 64 开始，所有请求的输出都和 HF 逐 token 相同，开不开重叠都测。
  - 按每次报 `Capacity` 就丢缓存的写法，这条失败；调度器不发恢复的拷贝，也失败。
- **其余**：调度器那组测试用的 worker 是夹具，不算状态，所以在它们的台子上关着缓存（`tests/units/scheduler/support.py`）。
- **恢复测试要只算几个 token**：随机权重下，GDN 状态几十个 token 就淡掉了，恢复以后算 36 个 token，什么都不恢复的对照也对得上。所以：
  - worker 的恢复测试改成恢复后算 4 个 token，并带上这个对照；
  - 端到端测试加了一个缓存后只剩 2 个 token 的请求，调度器不发恢复的拷贝时这条失败。

### 7.4 GPU（2026-11-13，RTX 4090 D，4 层随机权重）

在引擎上的过程见 [../rsi/2026-11-13-prefix-cache-on-a-live-engine.md](../rsi/2026-11-13-prefix-cache-on-a-live-engine.md)。

- **热改上去**：从 master 起的引擎，一次 apply 换上 #59，2.4 秒提交。
- **正确性**：fp32 下，命中和不命中逐字相同。bf16 下，存在 chunk 中间的状态是 bf16 的 `h`，和 SGLang 一样（它的 Triton GDN kernel 没有 fp32 快照），所以输出可能不同。
  - 存下的状态离单独算到那里的 1.5e-2（conv）、2.4e-2（temporal），错一个 token 的对照 1.4、1.3；
  - 恢复后算两个 token，状态离整个提示词的 2e-2，不恢复的对照 1.1。
  - 这三项是 `tests/gpu/test_prefix_cache.py`。它还查了 graph 回放里存下的状态就是那一行的槽，和 eager 的相同，没被选中的槽一位不动。
- **性能**：
  - 共享约 3000 token 系统提示，命中时首 token 27–35 ms，关掉缓存 102–122 ms；
  - 不命中时 decode 吞吐不变（169.2 / 1272.5 对 169.2 / 1270.4 token/s，1 和 8 并发）。
- **GPU 测试**：9 个文件 53 项全过，包括缓存默认开着的 `test_qwen38_serve`。

### 7.5 还没做的

- SM100 上的 64 层模型，以及和 SGLang 同配置（共享系统提示的负载）的对比。
- A/B/C。
