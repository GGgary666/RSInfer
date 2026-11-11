# Full attention 的 kernel 接入：现在的实际做法

- 日期：2026-11-11
- 状态：描述已经实现的行为，不是提案。
- 对应的代码：PR #51（分支 `real-kernels`）加 PR #53（分支 `exp/sm100-kv-store-indices`，提交 `f543495`）。**这两个 PR 合进 master 之前，本文提到的一部分文件在 master 上还不存在**，所以下面的路径只写成代码路径，不做链接。
- 范围：Qwen3.8、TP1 的 full attention 层。GDN、调度、状态分配不在这里。

本文最初是一份接入 FlashInfer 的设计提案（提交 `502f5ce`）。后来 #51 用另一种做法实现了，这里按实现重写；两者的差别列在最后一节。

## 1. 一句话概括

- bf16（或 fp16）模型、有 FlashInfer 的 GPU 上，attention 用 FlashInfer 的 paged kernel，eager 和 decode graph 是同一套。
- 其他情况（CPU、fp32、KV 的 dtype 和模型不同、没装 FlashInfer）用原来的 PyTorch 实现；这时 graph 里的 decode attention 是一个按最大上下文算的替身。
- 上面第一种情况下，attention 层里 attention 前后的小算子（Q/K norm、RoPE、拆 gate、KV 写入、输出 gate）各是一个 kernel，和 SGLang 在 CUDA 上跑的一一对应。

## 2. 一个 attention 层在一步里做什么

每步开始、任何 graph 之外，executor 让每个 backend 调一次 `prepare`。然后模型逐层前向。

`models/qwen38/unit.py` 的 `self_attention`：

| 顺序 | 做什么 | definition | 谁调的 |
|---:|---|---|---|
| 1 | QKV 投影（一次 GEMM），拆成"Q 和 gate 交错"、K、V | `gemm.linear@1` | 模型 |
| 2 | Q、K 按头做 GemmaRMSNorm，每个头的前 `rotary_dim` 维做 RoPE（Qwen3.8-27B 是前 1/4），把 gate 从 Q 里拆出来 | `attn.qk_norm_rope_gate@1` | `layers.rope` |
| 3 | 把这一步的 K、V 写进 KV 池 | `kv.store@1` | `layers.attn` 的 `forward` |
| 4 | attention | paged 一族或原有一族，见 §3 | `layers.attn` 的 `forward` |
| 5 | 输出乘以 `sigmoid(gate)` | `act.sigmoid_mul@1` | `layers.activation` |
| 6 | 输出投影 | `gemm.linear@1` | 模型 |

`prepare`（每步一次，`layers/attn/paged.py`）为 paged 一族准备两样东西：

- 每行的 KV 槽列表，和这一步要写入的槽（`attn.kv_indices@1`）；
- 让这一步的 operator 做 `plan`（见 §4）。

## 3. 两族 attention kernel，和怎么选

### 3.1 两族

| | 原有一族 | paged 一族 |
|---|---|---|
| definition | `attn.extend@1`、`attn.decode@1` | `attn.paged_extend@1`、`attn.paged_decode@1` |
| kernel 怎么知道每行的 K/V 在哪 | 自己读 `req_to_token` 表 | 每步先 `plan`：调用方把每行的槽列表交给它 |
| solution 是什么 | 一个函数 | 一个"造 operator 的东西"，operator 有 `plan` 和 `run`（§4） |
| 参照实现 | `kernels/reference/ops.py`：按行循环，SGLang 的 `TorchNativeAttnBackend` | `kernels/reference/paged.py`：同样的计算，结果逐位相同 |
| 真 kernel | 没有 | FlashInfer：`flashinfer.attn.paged_extend@1`、`flashinfer.attn.paged_decode@1` |
| 能不能录进 graph | 参照实现不能（在 host 上读长度）。graph 里用 `torch_graph.attn.decode@1`：每行都在整张表的宽度上算，靠长度 mask，代价跟 `batch.max_context_len` 走 | FlashInfer 的 decode 能；extend 不录 |
| backend | `TorchNativeAttnBackend`（`layers/attn/unit.py`） | `PagedAttnBackend`（`layers/attn/paged.py`） |

两族算的是同一个 attention。分成两族，是因为它们每步要准备的元数据不同。

### 3.2 选哪一族

worker 启动时，`core.plan` 让 `layers.attn` 的 `resolve` 按阶段（prefill、decode）各定一次，之后的步只调定下来的那些。配置键是 `attn.backend`，以及按阶段覆盖它的 `attn.prefill_backend`、`attn.decode_backend`。

选的时候带着一个 shape：

| 键 | 含义 |
|---|---|
| `head_dim` | 每个头的维度 |
| `fp32` | 模型或 KV 里有一个是 fp32 就是 1 |
| `kv_as_q` | KV 的 dtype 和模型的相同是 1 |

FlashInfer 的两个 attention solution 只接 `fp32 = 0` 且 `kv_as_q = 1`。原因：它不收 fp32；它把 16 位的 KV 直接当成 Q 的类型读，不做转换，bf16 模型配 fp16 的 KV 会算错。

- **`auto`（默认）**：paged 一族有非参照的 solution 接这个 shape，就用 paged；否则用原有一族。
- **写了名字**：这个名字有原有一族的 solution 就用原有一族（比如 `reference`），否则用 paged（比如 `flashinfer`）。两族都没有、或者 shape 不接，启动失败，报错里带着 shape。
- **graph 跟 decode 走同一族**，另外要求选到的 solution 可捕获。有一个不可捕获，就不录 graph，原因写进启动日志。
- **`kv.store@1` 和 `attn.kv_indices@1` 不属于某个 attention backend**（SGLang 里写 KV 是池自己的事）。点名的 backend 没有它们时，这两项按 `auto` 选。

常见的几种结果：

| 配置 | prefill 和 eager decode | graph 里的 decode |
|---|---|---|
| bf16，KV 也是 bf16，有 FlashInfer | FlashInfer paged | FlashInfer paged |
| fp32 | 参照实现 | `torch_graph` |
| bf16 模型，KV 是 fp16 | 参照实现（它会转换） | `torch_graph` |
| CPU | 参照实现 | 没有 graph |

启动时 `core.plan` 打一行 `core.plan: {json}`，里面有每个阶段每个 definition 实际选的 solution。

## 4. paged operator 的约定

写在 `kernels/reference/paged.py` 的文档里，FlashInfer 的实现（`kernels/flashinfer/ops.py`）照它来。

- **造**：`make(num_qo_heads, num_kv_heads, head_dim, q_dtype, kv_dtype, device, buffers)`。
  - eager 步：`buffers` 是 `None`，每个阶段一个 operator。
  - graph 桶：`buffers` 是这个桶的固定 buffer，每个桶一个 operator。
- **`plan(qo_indptr_cpu, kv_indptr_cpu, kv_indptr, kv_indices)`**：每次 `run` 之前、在 graph 之外调一次。
  - 第 `i` 行看的是 `kv_indices[kv_indptr[i]:kv_indptr[i+1]]` 这些槽；
  - 它有 `qo_indptr[i+1] - qo_indptr[i]` 个 query，是这一行最后的几个 token，各自按因果关系看前面的；
  - 带 `_cpu` 的是 host 上的那份。长度在 host 上本来就知道，所以 plan 不用等设备。
- **`run(q, k_cache, v_cache, scaling)`**：算出 `[tokens, heads, head_dim]`，eager 或者在 capture 里跑。

什么时候造、什么时候 plan：

- operator 在**第一次 `forward`** 时造，因为头数到那时才知道。那次 `forward` 也负责 plan。
- 之后每步在 `prepare` 里 plan。
- graph 桶的第一次 `forward` 是 capture 之前那次 eager 的预热，所以被录下来的 `forward` 里只有 `run`。
- 一个 operator 的 plan 里固定了头数。另一层用别的头数来调，直接报错。
- kernel 目录重新登记以后（kernel 单元被重载），旧的 operator 全部丢掉，下一步重新造。

## 5. KV 的布局、槽列表和写入

### 5.1 布局

- 每个 attention 层的 K、V 各是一个 `[槽数, kv_heads, head_dim]` 的张量，一个 token 一个槽（SGLang 的 `MHATokenToKVPool`）。存储按页分配（`state.kv.page_size`，默认 16 个 token），第一页是 padding。
- 持久 batch 有一张表 `req_to_token[行, 位置] = 槽号`。第 0 行是 padding 行。

### 5.2 槽列表：页大小按 1 算

交给 paged kernel 的列表里放的是 **token 槽号**，页大小告诉它是 1。这是 SGLang 的做法，好处是 KV 池的布局不用改。一行有多少 token，列表里就有多少项。

`attn.kv_indices@1`（#53）一次做两件事：

- 把每行在 `req_to_token` 里的前 `长度` 个槽，一行接一行抄进 `kv_indices`；
- 可选：把每行最后一个槽写进 `out_cache_loc`，那是 decode 步要写 K、V 的位置。

| solution | |
|---|---|
| `reference.attn.kv_indices@1` | torch 写的，不在 host 上读值，可以被录进 graph |
| `triton.attn.kv_indices@1` | SGLang 的 `create_flashinfer_kv_indices_triton`，多了"每行最后一个槽"这个输出，一个 kernel |

prefill 步一行不止一个新 token，`out_cache_loc` 直接从 `req_to_token` 里切出来。

### 5.3 KV 写入

`kv.store@1(k_cache, v_cache, loc, k, v)`：把这一步每个 token 的 K、V 写到 `loc` 给的槽里。

| solution | |
|---|---|
| `reference.kv.store@1` | K、V 各一次 `index_put` |
| `cuda.kv.store@1`（#53，`kernels/cuda/`） | 一个 C++/CUDA kernel，照 SGLang 的 `store_kvcache` 写。它不接的输入（K/V 和池的 dtype 不同、一行不是整数个 16 字节、没对齐）退回上面的写法 |

`kernels.cuda` 在单元加载时编译这个 kernel，一台机器上第一次约一分钟。改它的 C++ 源码再热改，要先在引擎外编译（`python -m kernels.cuda.ops`），否则变更会因为超时被撤回。

## 6. FlashInfer 的适配

`kernels/flashinfer/`，对应 SGLang 的 `--attention-backend flashinfer`。只在有 CUDA 设备的进程里 import 和登记。

- **版本**：`flashinfer-python` 0.7.2，SGLang `e838360` 钉的那个版本，在 `gpu` extra 里。
- **decode**：`BatchDecodeWithPagedKVCacheWrapper`。
  - 第一次 plan 用 FlashInfer 自己的 `begin_forward`，它会编译或加载 kernel 模块；
  - 之后用 `fast_decode_plan`，和 SGLang 给 graph 的 wrapper 装的一样。
- **extend**：`BatchPrefillWithPagedKVCacheWrapper`，在每行的整段序列上算，`causal=True`。新 token 的 K、V 先写进池，再算。这是 SGLang 不用 ragged wrapper 的那条路。
- **workspace**：每个设备一块 384 MiB（SGLang 的默认值），所有 wrapper 共用，归 `kernels.flashinfer` 单元所有。第一个 operator 造出来时分配。
- **tensor core**：照 SGLang 的规则，bf16 / fp16 下每个 KV 头对应的 Q 头不少于 4 个时用。Qwen3.8-27B 是 24 比 4，用。
- **编译**：第一次 plan 某个形状和 dtype 时要编译，`PATH` 上要有 `ninja` 和 CUDA 编译器。结果缓存在 `~/.cache/flashinfer`。

同一个单元还登记了两个不属于 attention 的 solution：`flashinfer.norm.gemma_rms@1` 和 `flashinfer.act.silu_and_mul@1`。

## 7. decode graph

`core/plan/`。只有 decode 步录 graph；prefill 和 mixed 步都是 eager。

### 7.1 桶

没配 `plan.graph_buckets` 时：

- graph 里的 attention 是 paged 的：1、2、4、8、12，然后每 8 行一档，到 `min(batch.max_rows, 64)`，并包含这个上限本身。这是 SGLang 的默认档位。
- graph 里的 attention 是 `torch_graph`：2 的幂，到 8。它的代价按最大上下文算，桶大了放不进预算。

每步 executor 选能装下这一步行数的最小的桶；没有桶装得下就走 eager（`plan.cuda_graph=required` 时在写任何状态之前拒绝这一步）。

### 7.2 每个桶固定的东西

`PagedAttnBackend` 为每个桶留一组固定 buffer，回放前在 `prepare` 里原地填：

| buffer | 大小 | 内容 |
|---|---|---|
| `kv_indptr` | 桶的行数 + 1，int32 | 每行的列表从哪开始 |
| `kv_indices` | 桶的行数 × `req_to_token` 的宽度，int32；所有桶共用一块，各取前面一段 | 槽列表 |
| `kv_last_page_len` | 桶的行数，int32 | 全是 1 |
| `out_cache_loc` | 桶的行数，int64 | 每条 lane 要写的槽 |

这些 buffer 的地址是 graph 的依赖。backend 把它们报给 `core.plan`（`binding`），地址换了，旧的 graph 就不能再回放。

### 7.3 padding lane

桶的行数比真实请求多时，多出来的 lane 是 padding：

- 它们被当成 **padding 行（第 0 行）、长度 1**，读写的都是 padding 槽。
- 几条 padding lane 会写同一个槽，谁最后写进去算谁的。没有真实请求读那里。
- padding lane 算出来的东西没有定义，采样结果被丢掉。
- GDN 层里 padding lane 的状态槽号是 -1，kernel 不为它读写状态。

因此 **padding 槽里的 K、V 可以是任何值**，包括非有限值。FlashInfer 只读每行列表里的槽，不受影响。`torch_graph` 的 attention 会读到 padding 槽（长度之外的位置都指向它），见 §9 的已知问题。

### 7.4 录制

`core.plan` 加载时（启动，或者热改时的暂停期内）：

1. 先跑一步 padding 行的 eager decode。库在这一步里分配它所有步共用的东西（FlashInfer 的 workspace、编译好的模块）。同时检查模型前向里用到的 solution 都可捕获。
2. 配置里没有 1 行的桶时，单独录一个 1 行的 graph 做自检，录完关掉，backend 释放为它留的东西。
3. 记下显存的起点，从大到小录每个桶，共用一个显存池。每个桶先 eager 预热一次，再 capture。
4. **自检只对 1 行的 graph 做**：回放的 logits 要等于 eager 的。多行的桶里有 padding lane，它们的输出没有定义，没有可比的行。
5. 录完留下的显存超过 `plan.graph_memory_bytes`（默认 1 GiB）：全部关掉。`auto` 退回 eager 并写明原因，`required` 启动失败。第 1 步分配的共用资源不算在这个预算里。

graph 关掉时（`core.plan` 卸载、预算超了、录制出错），backend 的 `release_graphs` 丢掉所有桶的 buffer 和 operator。

## 8. 热更新

- 改 kernel 单元或 `layers.attn` 的代码：`core.deps` 会连带重载 `core.plan`，graph 重录。`core.plan` 没有就地重建的动作。
- kernel 单元重载以后，`layers.attn` 手里的 operator 是旧代码的，下一步全部重新造。
- 没经过变更流程、目录却变了的情况（不应该发生），graph 的依赖检查会发现，回放之前拒绝旧图。
- 把整个 #51 第二批热改到第一批的引擎上、把 #53 热改到 #51 的引擎上，都试过，结果和冷启动一致（`doc/rsi/` 下的两份记录，在 #51 和 #53 里）。

## 9. 已知问题和没做的

- **`torch_graph` 的 attention 会被 padding 槽里的非有限值污染。** 它把长度之外的位置指向 padding 槽，只把分数 mask 掉，而 0 × NaN 还是 NaN，真实行的输出就坏了。本 PR 的历史里有一个两行的修复和 6 个回归测试（提交 `5e14f65`），后来随代码一起从这个 PR 里去掉了，**现在不在 master、#51、#53 的任何一个里**。细节见 [serving-flashinfer-gdn-ablation.md](../evaluations/serving-flashinfer-gdn-ablation.md)。
- extend 没有用 SGLang 默认的 ragged wrapper。
- prefill 不走 graph（SGLang 给 prefill 也录）。短 prompt 的首 token 时间因此压在 host 上。
- SGLang 在录 graph 时把 GDN 的一个小 GEMM 放到第二个 stream 上并行，这里没有。
- `cuda.kv.store@1` 里带 PDL 的那一支只在 SM90 以上的机器上编译，4090 D（SM89）上没跑过。
- FlashInfer 的 GDN kernel 和 `trtllm_mha` 没有接。

## 10. 测试在哪

- `tests/gpu/test_real_kernels.py`（要 GPU，手动跑）：
  - FlashInfer 的 operator 对照参照 operator，eager 和 graph 桶的固定 buffer 都测；
  - KV 写入、槽列表、Q/K norm + RoPE + gate、输出 gate 各自对照参照实现；
  - worker 栈（4 层、bf16）：`auto` 选的是真 kernel；graph 对 eager；KV 和模型 dtype 不同时不选 FlashInfer；被拒绝或只用来自检的 graph 不在 backend 里留东西；kernel 单元带着请求重载。
- `tests/units/executors/test_paged_attention.py`（CPU，进 CI）：CI 里没有库提供 paged kernel，测试把 backend 切到 paged 一族的参照 operator 上，整条 worker 栈的 token 和 logits 与原有一族相同。
- `tests/units/kernels/`：paged 参照和原有参照逐位相同；`attn.kv_indices@1` 的参照对着手写的列表查。

## 11. 和最初提案的差别

最初的提案（`502f5ce`）和实现在这几处选了不同的做法：

| | 提案 | 实现 |
|---|---|---|
| 接口 | 保留原 definition；给 kernel 目录加 attention 专用的 factory 和 session，新增一个契约文件；模型提前把每层的头数交给 backend | 新增两个 paged definition，solution 是造 operator 的东西；没动契约；头数在第一次 `forward` 时才知道 |
| 槽列表 | 页大小 16，列表里是物理页号 | 页大小 1，列表里是 token 槽号（SGLang 的做法） |
| padding lane 的 KV 写入 | 新增 `kv.store_masked@1`，只写真实行 | padding lane 写 padding 槽 |
| `auto` | 验收之前不选新 kernel | 合适就选 |
| workspace | 每个 operator 一块，graph 桶的算进 graph 预算 | 每个设备一块共用，不算进 graph 预算 |
| FlashInfer 版本 | 0.6.14 | 0.7.2 |
| 默认的 graph 桶 | 不动 | paged 时用 SGLang 的档位到 64 |
