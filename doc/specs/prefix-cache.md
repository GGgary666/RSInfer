# 前缀复用：照 SGLang 搬（计划，2026-11-13）

- 状态：计划，实现中（2026-11-13）。属于 M2d（[stage2-contract.md](stage2-contract.md) §1.3）。
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
| `_calculate_mamba_ratio` | `core.state.holder` | 开前缀缓存时，按同样的比例给出最大行数 |
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

### 2.4 显存怎么分（`core.state.holder`）

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
