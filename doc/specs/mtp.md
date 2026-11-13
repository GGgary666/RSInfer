# MTP 投机解码：照 SGLang 搬（计划，2026-11-13）

- 状态：计划；实现在同一个 PR 里（用户 2026-11-13 定）。属于 M2d（[stage2-contract.md](stage2-contract.md) §1.3）。
- 来源：SGLang `e838360`（`~/sglang-e838360/python/sglang/srt/`，下面的路径都相对于它）。Qwen3.5 结构的 MTP：`models/qwen3_5_mtp.py`，EAGLE 的 v2 worker（`speculative/eagle_worker_v2.py`、`eagle_worker_common.py`、`spec_utils.py`、`eagle_utils.py`），GDN 的验证路径（`layers/attention/linear/gdn_backend.py`、`hybrid_linear_attn_backend.py`），状态的提交（`kernels/ops/mamba/mamba_state_scatter_triton.py`）。
- 取代：stage2-contract.md §4.2、§4.4 第 7 条留给 M2d 的"草稿放在哪、验证步怎么数、开投机时位置怎么写"。

## 0. 用户定的（2026-11-13）

1. 只做链式：`(num_steps, topk, num_draft_tokens) = (3, 1, 4)`，SGLang 对 Qwen3.5 的默认。不做树（topk > 1），不搬这个 fork 加的东西（replayssm 环、自适应步数、post-verify graph、uno、dflash、PP / DP、双 batch 重叠）。
2. 开投机时也重叠（SGLang spec v2 的做法），不做只能同步的第一版。
3. 支持 temperature > 0。
4. 和前缀缓存一起用（验证步里照 SGLang 跟踪状态）。
5. 默认关，`spec.algorithm` 打开。
6. 计划和实现一个 PR。
7. 不做 A/B/C。

## 1. SGLang 怎么做

### 1.1 草稿模型（`Qwen3_5ForCausalLMMTP`）

- `fc(cat(pre_fc_norm_embedding(embed(x)), pre_fc_norm_hidden(h)))`，`fc` 是 `2 * hidden -> hidden` 的线性层；然后一层 full-attention 的 decoder 层（`mtp_num_hidden_layers = 1`，`full_attention_interval = 1`），最后 `norm`，`lm_head`。
- `embed_tokens`、`lm_head` 和目标模型共用（`mtp_use_dedicated_embeddings = false`；`set_embed_and_head`）。
- `h` 是目标模型最后一层 norm 之后的 hidden（`LogitsProcessor` 存的 `hidden_states`）。
- 权重在同一份 checkpoint 里，名字是 `mtp.*`：`mtp.fc`、`mtp.pre_fc_norm_*`、`mtp.layers.0.*`、`mtp.norm`；加载时 `mtp.` 换成 `model.`（`load_weights`）。我们的目标模型加载时把它们跳过了（`engine/models/qwen38/unit.py`）。
- 它自己的一层 attention 有自己的 KV，和目标模型用同一张页表、同一套 token 槽（`draft_runner` 和目标共用 `req_to_token_pool` 和 allocator），所以前缀缓存的页里也有它的 KV。

### 1.2 一个 decode 步（`EAGLEWorkerV2.forward_batch_generation`）

记一行已经确定的长度为 `L`（KV 里有位置 `0 .. L-1`），上一步接受的最后一个 token（bonus）`b` 在位置 `L`，还没进 KV；上一步的草稿扩展给了第一个草稿 `d1` 和草稿模型在那里的 hidden。D = `num_draft_tokens` = 4，S = `num_steps` = 3。

1. **草稿**（`draft`、`draft_forward`）：草稿模型跑 S - 1 = 2 次 decode：输入 `d1` 在位置 `L`，得到 `d2`；输入 `d2` 在 `L + 1`，得到 `d3`。每次用上一次的输出 hidden。topk = 1：取 argmax（temperature > 0 也一样，见 1.4）。写草稿模型的 KV 在 `L`、`L + 1`。
2. **验证**（`run_eagle_verify`）：目标模型一次算 `[b, d1, d2, d3]`，位置 `L .. L+3`，写目标 KV；attention 是每行 4 个 token 的 causal extend（topk = 1 不要树 mask）。四个位置都出 logits 和 hidden。GDN 不改请求自己的状态（1.3）。
3. **接受**（`eagle_sample`）：`p_j` 是位置 `L + j` 之后的 token（j = 0..3）。贪心时 `d_{j+1} == argmax p_j` 就接着比，第一个不同的地方停；接受长度 `a`（1..4）包括最后改正的或者奖励的那个 token。输出 `p_0 .. p_{a-1}`。新长度 `L + a`，新的 bonus 是 `p_{a-1}`。
4. **提交 GDN 状态**（`commit_mamba_states_after_verify`）：第 `a - 1` 步的中间状态拷回请求的槽；跨过前缀缓存的跟踪边界的，那一步的中间状态拷进跟踪槽（1.5）。
5. **草稿扩展**（`_draft_extend_for_decode`）：草稿模型在 `L .. L+3` 四个位置上算一遍，输入是 `p_0 .. p_3`，hidden 是验证时目标模型的四个 hidden。它的 KV 写在 `L .. L+3`（`a` 之后的那些以后会被重写）。取第 `a - 1` 行：它的 argmax 是下一轮的 `d1`，它的 hidden 是下一轮草稿的输入。

一个步写的位置都在 `[L, L + D)` 里（`prepare_for_draft` 说"plans the iteration's whole write window"）。

**prefill**（`_forward_prefill_batch`）：目标模型照常 prefill，留下整段 hidden；草稿模型在这个 chunk 上做一次扩展（`_draft_extend_for_prefill`），输入错开一位：位置 `i` 的输入是 `x_{i+1}`，chunk 的最后一个位置是下一个 chunk 的第一个 token，或者（结束提示词时）这一步采样出的 token。结束提示词的行由此得到 `d1` 和 hidden。

### 1.3 GDN 在验证步里

- conv：`causal_conv1d_update` 带 `intermediate_conv_window`：每一步之后的窗口写进中间缓存，不改请求的 conv 状态。
- 递归：`fused_sigmoid_gating_delta_rule_update`（`disable_state_update=True`、`intermediate_states_buffer`、`cache_steps=D`）：从请求的状态出发，每一步之后的状态写进 `intermediate_ssm`，不改请求的状态。
- 中间缓存的形状：SSM `[GDN 层, 行 + 1, D, HV, V, K]`，conv `[GDN 层, 行 + 1, D, conv_dim, K - 1]`。48 个 GDN 层时一行约 4 × 147 MiB = 590 MiB。
- 提交（`scatter_mamba_states_after_mtp_verify`）：第 `a - 1` 步的中间状态拷到请求的槽；`mamba_steps_to_track >= 0` 的行，那一步的拷到跟踪槽。

### 1.4 temperature > 0

CUDA 上默认不开 `speculative_use_rejection_sampling`，用 target-only 的投机采样（`tree_speculative_sampling_target_only`）：草稿照样取 argmax，接受与否只看目标模型的概率（先按 temperature、top-k、top-p 处理）：第 `j` 个草稿以 `p_j(d_{j+1})` 的概率被接受；第一个被拒的位置从去掉这个草稿以后重新归一的分布里采；全部接受时从 `p_{D-1}` 采奖励的那个。随机数照我们的规矩从 `(seed, position)` 取（`SamplingParams.seed`），所以结果和怎么分批无关。

### 1.5 和前缀缓存

- `mamba_track_interval >= num_draft_tokens`（`mamba_hook`），所以一个验证步最多跨一个跟踪边界。
- 开投机时每个 decode 行每一步都带跟踪槽（`mamba_track_indices`）；跨没跨、在哪一步，在设备上按接受长度算（`_verify_commit_step_indices`：`L // grid != (L + a) // grid`，步号 `(L + a) // grid * grid - L - 1`）。

### 1.6 重叠（spec v2）

- 调度器在第 N 步的结果回来之前就排第 N + 1 步，不知道第 N 步接受了几个。
- KV 按最坏情况给：已知的长度往后 `2 × D`，按页取整（`get_alloc_reserve_per_decode`、`page_aligned_decode_alloc_lens`）。
- 第 N + 1 步的位置在设备上：第 N 步验证完把新长度写出来（`FutureMap`、"Publish before draft_extend"），第 N + 1 步要在 host 上做 attention 的 plan 时，等这个事件，把新长度拷回 host（`resolve_seq_lens_cpu`）。这时设备还在跑第 N 步的草稿扩展。
- 开投机时不做混合步（`enable_mixed_chunk = False`）。

### 1.7 graph

草稿（S - 1 次 decode 加 argmax）、验证（每行 D 个 token 的目标 forward）、草稿扩展（每行 D 个 token 的草稿 forward 加取行）各录一组，按行数分桶（`eagle_draft_cuda_graph_runner.py`、`eagle_draft_extend_cuda_graph_runner.py`、验证用目标的 decode runner，`num_tokens_per_bs = D`）。

## 2. 我们这边怎么对上

### 2.1 配置

`executors.pytorch` 声明，用到的单元在 `[reads]` 里读：

| 键 | 默认 | SGLang |
|---|---|---|
| `spec.algorithm` | `""`（关） | `--speculative-algorithm`；开就是 `"eagle"`（MTP 是 SGLang 的 NEXTN，按 EAGLE 跑） |
| `spec.num_steps` | 3 | `--speculative-num-steps` |
| `spec.eagle_topk` | 1 | `--speculative-eagle-topk`，只收 1 |
| `spec.num_draft_tokens` | 4 | `--speculative-num-draft-tokens`，必须是 `num_steps + 1` |

都是 startup 绑定：在空引擎上 apply，用到的单元重载。

### 2.2 接口（`core.contracts`，原地改，带默认值，旧记录照样能解）

- `BatchPlan.draft_tokens: int = 0`：开投机时是 D。这时：
  - decode 行验证 D 个 token：`num_tokens` 仍是 1（输入的那一个是 bonus，`from_previous`），草稿是 worker 自己的；
  - 有 `from_previous` 的 decode 行，位置以 worker 记的为准，`positions` 只是调度器知道的下限（它前面的步可能还在路上）；
  - prefill 行另外做草稿扩展；
  - 不混合 prefill 和 decode。
- `StateTrack` 开投机时的 decode 行：`upto = -1`，"这一步跨过的跟踪边界，如果跨了"；跨没跨 worker 在设备上算，调度器在步退役时按接受长度算出同样的结果。
- `Sampled.counts`：开投机时 decode 行是接受长度 `a`（本来就这么写的）。行前进的就是 `a`。
- 账本（`ledger@1`）：`retire(step, ran, advanced)` 多一个参数，开投机时每个 decode 请求实际前进了多少；`reserve` 照常给页。

### 2.3 调度器（`core.sched`）和请求表（`sched.reqs`）

- 开投机时一个 decode 行的 KV 预留到 `computed + (在路上的步数 + 1) × D`（SGLang 的 2 × D）。
- 接纳时要能放下 `prompt + max_tokens + 2 × D`。
- 在路上的步按 D 个 token 算（最坏），退役时按 `Sampled.counts` 前进（请求表的 `retire` 收实际的前进）。
- 有 prefill 也有 decode 时，这一步只排 prefill（不混合）。

### 2.4 worker

- **模型**（`models.qwen38`）：开投机时，assembly 里多草稿模型的参数（`mtp.*` 照 SGLang 的映射）和它那一层的 KV（`kv.paged` 的第 `num_hidden_layers` 层，和目标共用页号，holder 一起算），并另外提供 `model.draft@1`。目标的 `forward` 能返回所有 token 的 logits 和 norm 之后的 hidden。
- **持久 batch**（`worker.batch`）：每行多几张表：已确定的长度（设备上，和 host 上的镜像）、草稿 token、草稿模型的 hidden、有没有草稿。graph 录它们的地址。
- **GDN 存储**（`state.gdn_slots`）：开投机时多中间缓存，按 `batch.max_rows` 行，holder 先从预算里扣掉。
- **GDN 后端**（`layers.gdn`）：验证步用带中间缓存的 conv 和递归；提交是另一个 kernel。
- **executor**（`executors.pytorch`）：投机的一步放在 `spec.py`：草稿、验证、接受、提交、草稿扩展；prefill 步后接草稿扩展。验证完记一个事件，把接受长度和 token 拷进 pinned 内存；下一步要位置时先等这个事件（1.6）。
- **kernel**（`kernels.catalog`）：新定义，每个都有 PyTorch 参照：
  - `gdn.verify_conv1d_update@1`、`gdn.verify_recurrent@1`（SGLang 的 Triton 版）；
  - `gdn.commit_states@1`（SGLang 的 `fused_mamba_state_scatter_with_mask`）；
  - `spec.accept@1`：贪心和 target-only 采样，torch 写，不和 host 来回，能录进 graph。
- **graph**（`core.plan`）：草稿、验证、草稿扩展三组，按行数分桶；接受和提交先不录。

### 2.5 checkpoint 和部署

- `deploy/agxinfer.py checkpoint` 能写随机的 `mtp.*`（给本机的 4 层用）。
- `deploy/agxinfer.py serve --speculative-algorithm eagle`（和 SGLang 同名）。

## 3. 和 SGLang 不一样的地方（预期的）

- 草稿模型在目标模型的单元里（一个单元提供 `model@1` 和 `model.draft@1`），不是另一个 model runner：它们共用权重存储、KV 存储和 holder 的预算，开关投机就是重载这个单元。
- 位置由 worker 记：调度器不知道在路上的步接受了几个，worker 等上一步验证完的事件。SGLang 是调度器那边的 `FutureMap`。
- 持久 batch 重建（`Resync`）以后，一行没有草稿：它这一步用占位的草稿、强制只接受 1 个（等于一个普通的 decode 步加草稿扩展），下一步起照常。SGLang 没有 `Resync`。

## 4. 验收

- **CPU（进 CI）**：
  - 参照 kernel：验证步的 conv、递归和单独一步步算的相同；提交拷对了步。
  - "预言草稿"的测试夹具：草稿就是目标模型的贪心 token，在第 j 个位置故意错一个，接受长度 1..4 都走到；每种之后 GDN 状态、logits 和不开投机一步步 decode 的相同（fp32）。随机的草稿权重几乎全拒绝，覆盖全拒。
  - 端到端：开投机和不开，贪心 fp32 逐 token 相同，开不开重叠、开不开前缀缓存都测。这是五组对比的最后一组。
  - temperature > 0：同一个 seed，换分批方式结果相同；分布上和不开投机一致（统计检验）。
- **4090 D**：真 kernel、graph，bf16 按 stage2-contract.md §1.3 的口径；吞吐和不开投机比。
- **SM100**（Devin 或 Codex）：真 MTP 权重的接受长度和吞吐，和 SGLang 同配置比。

## 5. 分几步做（都在这一个 PR 里）

1. 接口、调度器、请求表、账本；夹具 worker 测。
2. 模型（草稿模型、权重、KV）、参照 kernel、executor 的投机一步（eager）、持久 batch 的表；CPU 对照测试。
3. temperature > 0。
4. 重叠。
5. 前缀缓存。
6. GPU：Triton kernel、FlashInfer 验证步的 metadata、三组 graph；4090 D 验收。
7. 文档、skill、RSI 记录。
