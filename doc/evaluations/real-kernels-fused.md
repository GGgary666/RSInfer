# 真 kernel 第二批：融合路径，和 #51 review 的修复（2026-11-11，RTX 4090 D，4 层）

接着 [real-kernels-4090d.md](real-kernels-4090d.md)（第一批：Triton GDN 和 FlashInfer attention）。这一批做两件事：

- 把每层里剩下的小算子换成 SGLang 在 CUDA 上实际跑的那些融合 kernel；
- 修 Codex 和 Gemini 在 #51 上提的问题。

起因是 Gemini 在 SM100 上用完整的 64 层模型测到 decode 只有 SGLang 的 64–72%，prefill 74%，而这台 4090 D 上的 4 层模型是 96%。差距随层数放大，来自每层多出的 kernel launch。

环境、checkpoint、测法都和第一批相同。SM100 上的结果这里量不了，等 Gemini 重测。

## 结论

1. **graph 回放的一步 decode，kernel launch 从 238 次降到 89 次，SGLang 是 82 次**（eager 的一步是 85 次）。 GDN 层里的 kernel 和 SGLang 一一对应，每层 11 个（5 个 GEMM、2 个 norm、SiLU·mul、conv、delta rule、gated norm）。attention 层在回放的步里是 13 个，SGLang 是 12 个：多的一个是 KV 写入（这里是两次 `index_put`，它是一个 kernel）。其余的差别是每步一次的准备工作，不随层数增长。
2. **这台卡上，经 HTTP 的 decode 吞吐是 SGLang 的 98.5–100.6%**（第一批是 95.5–98.9%），**prefill 是 94.6–99.4%**（第一批 94%）。1 并发的 prefill 仍然慢 8 ms（70 对 62 ms），原因在 host 一侧，见下。
3. **prefill 一步（2000 token）的 launch 从 311 次降到 162 次，比 SGLang 的 257 次少**；设备上的 kernel 时间 47.5–49.2 ms，和 SGLang 的 47.2 ms 持平（差别在 GEMM 自己的波动之内）。
4. **review 提的问题都成立，都修了，各有回归测试。** 其中一个是静默的数值错误：bf16 模型配 f16 的 KV 时 FlashInfer 的输出完全不对（我在 GPU 上复现：和参照实现最大差 816）。
5. **对 SM100 的预期没有验证过。** 按 Gemini 的归因（差距来自每层未融合的小算子和 launch 间隙，每层约 63 µs），每个 GDN 层多出的 launch 现在是 0，每个 attention 层是 1，decode 应该接近 SGLang；这是推算，要以重测为准。

## review 的修复

| 问题 | 谁提的 | 修法 | 测试 |
|---|---|---|---|
| FlashInfer 把 16 位的 KV 直接当成 Q 的 dtype 读，`auto` 在 bf16 模型 + f16 KV 时仍选它 | Codex | attention 的 shape 多一项 `kv_as_q`（KV 的 dtype 是否等于模型的）；FlashInfer 的两个 attention solution 只接 `kv_as_q = 1`。`auto` 在不相等时退回会做转换的原有 kernel；点名 `flashinfer` 时启动失败 | CPU：`test_config.py::test_the_attention_kernels_are_asked_for_with_the_dtypes_of_queries_and_cache`；GPU：`test_a_cache_of_another_16_bit_dtype_than_the_model_is_not_flashinfers` |
| FlashInfer 的 384 MiB workspace 在 graph 显存的统计窗口里第一次分配：冷启动算进 `plan.graph_memory_bytes`，只重载 `core.plan` 时不算，同一份配置两种结果 | Codex、Gemini | 录图之前先跑一次 padding 行的 eager decode 步，库在这一步里分配它所有步共用的东西；之后才取统计的起点 | GPU：`test_what_the_graphs_keep_is_counted_alike_cold_and_after_a_reload`（预算 200 MiB，比 workspace 小，冷启动照样录；重载后的 `kept_bytes` 和冷启动相差不超过 16 MiB） |
| 配置里没有 1 行的桶时，为自检录的那个 1 行 graph 关掉以后，两个 backend 里还留着它的元数据、固定 buffer 和 FlashInfer wrapper | Gemini | 自检的 graph 关掉后让 backend 释放它们（这时还没有录别的桶） | GPU：`test_graphs_that_are_refused_or_only_checked_leave_nothing_in_the_backends` |
| `gdn.conv1d_fn@1` 的 bf16 容差 1e-2（绝对加相对）容得下一个 bf16 舍入单位，容不下两个：SM100 上 270 万个元素里有 1 个在 2.09 附近差两个单位，0.03125，容差是 0.0309 | Gemini | bf16 结果的容差改成 2e-2（任何大小的值上都容得下两个舍入单位），理由写在 `kernels/reference/unit.py` | 原测试 |
| "graph 等于 eager"的测试拿 3 行的 eager 和落进 4 行桶的 graph 逐值比；SM100 上 cuBLAS 给 3 行和 4 行选了不同的 GEMM kernel，logits 差 0.117（token 相同） | Gemini | 拆成两个测试：桶是 1、2、3、4 时两边行数相同，逐值比；带 padding 行的桶比方向（余弦）和 token，eager 的前两名相差不到 0.5 时允许分开 | GPU：`test_auto_runs_the_real_kernels_and_a_replayed_step_is_the_eager_step`、`test_a_padded_bucket_gives_the_real_rows_what_they_get_alone` |
| 默认的 graph 桶只到 8 行；SM100 上 16–64 并发退回 eager 后慢 2.7–3.3 倍 | Gemini | 见下面"默认的桶" | CPU：`test_cuda_graph.py::test_default_buckets_and_required_coverage` |
| `bench_serving.py`：只在事件带文本时记时间，会低估每 token 间隔；keep-alive 连接数没跟并发数走 | Gemini | 每个 token 的事件都记；`max_keepalive_connections` 设成并发数 | 重测了 SGLang：数字没变（这个 tokenizer 下没有空文本事件） |

两点和 review 的说法不完全一样的地方：

- **workspace 在 graph 被拒绝以后还在，这不算泄漏。** 它是 `kernels.flashinfer` 单元的，eager 步也用它；graph 被拒绝以后跑的正是 eager 步。原来错的是把它算进了 graph 的预算，以及冷启动和重载的结果不一样。`test_cuda_graph.py` 里"拒绝以后显存回到原值"是在 worker 关掉以后量的，那时 workspace 随单元一起释放。
- **自检的那次 capture 还在**，所以配了 3 个桶时 `recorder.capture()` 仍然是 4 次。这是设计（第一批文档"三处行为变化"第 2 条），修的是它留下的东西。

### 默认的桶

没配 `plan.graph_buckets` 时：

- **graph 里的 attention 按行的长度算代价时**（paged decode）：用 SGLang 的默认档位 1、2、4、8、12，然后每 8 行一档，到 `min(batch.max_rows, 64)`，并且把这个上限本身也录上。这样 `max_rows` 不超过 64 时任何行数的 decode 步都有 graph，`required` 在默认配置下也满足"覆盖 `batch.max_rows`"；超过 64 时更多行的步走 eager，`required` 仍要显式列桶。`max_rows` 32 时是 1、2、4、8、12、16、24、32，和 SGLang 录的一样，`kept_bytes` 128 MiB。
- **graph 里的 attention 是固定容量的那个时**（`kernels.torch_graph`：fp32、KV 和模型 dtype 不同、或没有 FlashInfer）：仍是 2 的幂到 8。它每行要搬 `max_context` 那么长的 K/V，fp32 下 64 行 × 4096 时光是 K 的 gather 就是 1.07 GB，超过 1 GiB 的默认预算；而预算是"全录或全不录"，放开上限会让小桶也录不成。
- `core.plan` 怎么知道是哪一种：录进 graph 的 solution 里有没有只给 graph 用的（`requires` 含 `CAPTURE`）。

## 融合路径

### 换了什么

| 位置 | 之前 | 现在（SGLang 的对应物） |
|---|---|---|
| 残差上的 RMSNorm，每层 2 次，加末层 1 次 | 参照实现，每次约 10 个小 kernel | FlashInfer 的 `gemma_fused_add_rmsnorm` / `gemma_rmsnorm`，每次 1 个（`GemmaRMSNorm._forward_impl`）。带 residual 时原地改 |
| attention 层：Q/K norm、RoPE、拆出输出 gate | 2 次 norm + RoPE 的 21 个小 kernel + 几次 split / chunk | 1 个 Triton kernel（`fused_qk_gemma_rmsnorm_rope_gate`，`forward_prepare_cuda_fused`） |
| attention 层的输出 gate | `x * sigmoid(gate)`，2 个 kernel | 1 个 Triton kernel，原地改（`fused_sigmoid_mul`） |
| MLP 的 SiLU·mul | 参照实现 2 个 kernel | FlashInfer 的 `silu_and_mul` 1 个（SGLang 用的是它自己 JIT 编译的同一个 kernel） |
| GDN decode：拆两个输入投影 + conv | 模型里 split、reshape、cat、两次 contiguous，再加 conv kernel | 1 个 Triton kernel（`fused_qkvzba_causal_conv1d_update_contiguous`） |
| GDN decode：delta rule | 先 split 出 q、k、v，再跑 recurrent kernel | 1 个 Triton kernel 直接读拼在一起的 q、k、v（`packed_decode`） |
| GDN prefill：拆两个输入投影 | 模型里 split、cat、拷贝 | 直接用投影的 strided view（`qwen3_5_gdn_prefill_projection_views`），不拷贝 |

搬来的文件（`engine/kernels/triton/`，文件头注明出处，kernel 本体没动）：

| 这里 | SGLang `e838360` 的 `python/sglang/kernels/ops/` |
|---|---|
| `fused_qk_rmsnorm_rope_gate.py` | `attention/fused_qk_rmsnorm_rope_gate.py`，整个文件 |
| `triton_gdn_fused_proj.py` | `attention/triton_gdn_fused_proj.py`，整个文件，AMD 的开关固定成关 |
| `fla/fused_recurrent.py` | `attention/fla/fused_recurrent.py` 里的 packed decode kernel 和它的 launcher |
| `elementwise.py` | `elementwise/elementwise.py` 里的 `fused_sigmoid_mul` 和它的 kernel |

`fused_sigmoid_gating_recurrent.py`（不走 packed decode 时的 recurrent kernel）删掉了，没有调用方了。

### definition 的变化

| definition | 变化 |
|---|---|
| `attn.qk_norm_rope_gate@1` | 新。取代 `pos.rope@1`（删掉）。参照实现是 `forward_prepare_native` 那几步的组合 |
| `act.sigmoid_mul@1` | 新。调用方交出 `x`：solution 可以原地写 |
| `gdn.qkvzba_conv1d_update@1` | 新。取代 `gdn.conv1d_update@1`（删掉）。参照实现是"取 view + 原来的 conv 参照" |
| `gdn.packed_decode@1` | 新。取代 `gdn.recurrent_update@1`（删掉）。参照实现是"split + 原来的 recurrent 参照" |
| `norm.gemma_rms@1` | 签名不变；容差从逐位改成有容差；写明带 residual 时调用方交出 `x` 和 `residual` |
| `act.silu_and_mul@1` | 签名不变；容差从逐位改成有容差 |
| `norm.rms_gated@1` | 签名不变；`z` 可以是按 token 和头排的三维 view（prefill 时就是投影的 view） |

- 被取代的三个 definition 删掉了，因为层不再调用它们；它们的参照函数还在，是新参照的组成部分，`tests/units/kernels/test_reference.py` 仍然拿它们对照 HF。
- 新的参照实现在 CPU 上经整个 worker 栈对照 HF，逐 token 相同（原有的测试，没有改）。
- **哪个 solution 接哪种输入**（都靠 shape 里的一项来选，不合适就退回参照实现）：
  - FlashInfer 的 SiLU·mul、Triton 的融合 Q/K kernel：只接 bf16 / fp16（`fp32 = 0`）；
  - Triton 的融合 conv：conv 状态的 dtype 要和激活的相同（`conv_as_x = 1`）。fp32 模型配默认的 bf16 conv 状态时走参照实现。SGLang 在这种情况下退回它不融合的 Triton kernel，这里没有搬那条退路；
  - FlashInfer 的 RMSNorm、Triton 的 packed decode、sigmoid·mul、gated RMSNorm：fp32 也接。

### 接口的变化

- **`layers.gdn@1` 的 `forward`**（`core/contracts/plan.py` 的 `GDNBackend`）：原来接模型拆好的 `mixed_qkv`、`a`、`b`，返回 core 输出；现在接两个输入投影（GEMM 的原样输出），返回 `(core 输出, z)`。拆投影挪进了 backend，因为 decode 和 prefill 的拆法不同，而知道这一步是哪个阶段的是 backend。这是进程内的调用接口，不是消息；调用方只有 `models.qwen38`，一起改了。
- `layers.rope@1` 的 `forward` 换成 `qk_norm_rope_gate`；`layers.activation@1` 多了 `sigmoid_mul`。
- `models.qwen38` 的 `self_attention`、`linear_attention` 照 SGLang 的 CUDA 路径改写。

框架本体（`engine/framework.toml` 列的模块）没有动。

## launch 数和设备时间

### decode（8 行，每步）

按来源分的 launch 数是在 eager 下给每个 definition 的调用打标签量的；SGLang 一栏是按它 trace 里的 kernel 名字归的（graph 回放）。

| 来源 | SGLang | 只换 RMSNorm | 融合以后 |
|---|---:|---:|---:|
| GEMM（20 次，其中 3 次带一个 cuBLAS 的辅助 kernel） | 23 | 23 | 23 |
| RMSNorm（残差上 9 次；融合之前 Q/K 的 2 次也在这里） | 9 | 11 | 9 |
| RoPE（融合以后是 Q/K norm + RoPE + 拆 gate 的那一个 kernel） | 1 | 21 | 1 |
| SiLU·mul | 4 | 8 | 4 |
| GDN 三层的 conv、delta rule、gated norm | 9 | 9 | 9 |
| 模型里 kernel 之间的 torch 操作（拆投影、拼接、输出 gate、取每行最后一个 token） | 1（输出 gate） | 18 | 2 |
| KV 写入 | 1 | 2 | 2 |
| attention | 2 | 1（回放时 2） | 1（回放时 2） |
| 采样 | 3 | 3 | 3 |
| 每步一次的准备（各 backend 的 `prepare`、输入和状态的拷贝、embedding） | 29 | 31 | 31 |
| 合计 | 82 | 127 | 85 |

- SGLang 的"每步一次的准备"是合计减去其余各行得到的，里面有 1 次是拼 KV 槽列表的 kernel。
- graph 回放时量到的是 89 次（第一批 238，其中参照实现的 RMSNorm 一项约 100 次；只换 RMSNorm 131），比 eager 多 4 次。其中 1 次是 attention：给 graph 用的 FlashInfer wrapper 总是带一个合并分块结果的 kernel（SGLang 的回放里也有），eager 的 wrapper 在这个长度下不分块。其余净 3 次是回放前往固定 buffer 里拷贝，比 eager 现造张量多出来的。
- 所以回放的步和 SGLang 比是 89 对 82：每个 attention 层多 1 次（KV 写入），每步一次的准备多 6 次。
- 设备上 kernel 时间合计（graph 回放）：6.37 → 6.22 → 6.15 ms，SGLang 是 6.22 ms。其中 GEMM 5.94–5.96 ms。

### prefill（一步 2000 个 token，eager）

| | SGLang | 第一批 | 融合以后 |
|---|---:|---:|---:|
| launch 次数 | 257 | 311 | 162 |
| 设备上 kernel 时间合计 | 47.2 ms | 51.3–51.5 ms | 47.5–49.2 ms（2 次） |
| 其中 GEMM | 43.2 | 43.4–44.2 | 44.0–45.7 |

GEMM 之外的部分从 7–8 ms 降到 3.5 ms。两次测量之间 GEMM 自己差了 1.7 ms，合计的差别在这个波动之内。

## 和原生 SGLang 对比（重测）

同一天、同一个 checkpoint、同一个 tokenizer、同一版 `tests/gpu/bench_serving.py`，两边先后各跑一遍。

- SGLang：`e838360` 默认配置，命令和第一批相同（`--attention-backend flashinfer --context-length 4096 --max-running-requests 32`）。
- RSInfer：`deploy/qwen38.py serve --cuda-graph auto --overlap --max-rows 32 --max-context 4096 --token-budget 4096`。graph 桶用默认的，就是 SGLang 录的那 8 档。

decode（prompt 32 token，生成 256 个，每个客户端 4 个请求）：

| 并发 | SGLang token/s | RSInfer token/s | 比例 | 第一批的比例 | SGLang 每 token | RSInfer 每 token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 171.0 | 168.7 | 98.7% | 96.4% | 5.80 ms | 5.86 ms |
| 4 | 656.8 | 647.2 | 98.5% | 95.5% | 6.01 ms | 6.12 ms |
| 8 | 1276.7 | 1272.2 | 99.6% | 96.1% | 6.15 ms | 6.18 ms |
| 16 | 2426.3 | 2415.7 | 99.6% | 98.9% | 6.43 ms | 6.44 ms |
| 32 | 4161.7 | 4188.0 | 100.6% | 96.9% | 7.35 ms | 7.28 ms |

prefill（prompt 2000 token，生成 8 个，每个客户端 8 个请求）：

| 并发 | SGLang token/s | RSInfer token/s | 比例 | 第一批的比例 | SGLang 首 token | RSInfer 首 token |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19235 | 18187 | 94.6% | 93.5% | 62 ms | 70 ms |
| 4 | 31449 | 31117 | 98.9% | 94.5% | 162 ms | 171 ms |
| 8 | 35861 | 35640 | 99.4% | 93.9% | 257 ms | 265 ms |

- SGLang 这次的数字和第一批那次相差不到 3%：16 并发的 decode 是 2426 对 2366，4 并发的 prefill 是 31449 对 30801，其余在 1% 以内。
- decode 的首 token 时间：SGLang 17 / 25 / 31 / 45 / 81 ms，RSInfer 22 / 20 / 31 / 49 / 86 ms。
- **1 并发的 prefill 每个请求仍然多 8 ms。** 设备上的 kernel 时间已经持平，launch 还比 SGLang 少，所以这 8 ms 不在设备上。RSInfer 的 prefill 是 eager，每步 162 次 launch 从 Python 逐个发出；SGLang 的 prefill 是从它的 prefill graph 回放的。并发高了以后设备一直有活，这一项被盖住。具体是 host 上的哪一段，我没有再拆。

进程内的 microbench（`tests/gpu/bench_cuda_graph.py --max-context 256 4096 --repeats 2`，每步毫秒，`max_context` 4096）：

| batch | depth | 方式 | 第一批 | 只换 RMSNorm | 融合以后 |
|---:|---:|---|---:|---:|---:|
| 1 | 1 | eager | 7.61 | 7.16 | 6.95 |
| 1 | 1 | graph | 6.80 | 6.70 | 6.66 |
| 1 | 2 | eager | 6.13 | 5.94 | 5.90 |
| 1 | 2 | graph | 6.00 | 5.90 | 5.87 |
| 4 | 1 | eager | 8.15 | 7.52 | 7.25 |
| 4 | 1 | graph | 7.18 | 7.09 | 7.00 |
| 4 | 2 | eager | 6.46 | 6.26 | 6.16 |
| 4 | 2 | graph | 6.32 | 6.19 | 6.13 |
| 8 | 1 | eager | 8.19 | 7.74 | 7.46 |
| 8 | 1 | graph | 7.41 | 7.24 | 7.18 |
| 8 | 2 | eager | 6.55 | 6.33 | 6.23 |
| 8 | 2 | graph | 6.40 | 6.25 | 6.19 |

- 8 行、graph、depth 2 的吞吐 1250 → 1293 token/s；graph 和 eager 的 token 12 组配置里 12 组相同。
- 8 行 eager 一步在 host 上的提交时间 5.2 → 3.3 ms（launch 少了）。

## 和 HF 的对照

`tests/gpu/test_qwen38_partial.py`，4 层：

- fp32：eager、graph 的贪心 token 和 HF 12 步全部相同（三个 prompt 都是 12）。
- bf16：prefill logits 和 HF 的余弦 0.99940–0.99958。贪心 token 和 HF 相同的步数，三个 prompt 是 12、1、12，eager 和 graph 相同。第一批是 12、10、2，只换 RMSNorm 时是 11、1、2。随机权重下候选很接近，每换一个舍入方式不同的 kernel，分开的位置都会变；按 §1.3 的口径，bf16 只报告分歧位置。

## 在跑着的引擎上热改

`deploy/qwen38.py serve`（4 层、bf16、`--cuda-graph auto --overlap --max-rows 8`），每次变更前后用同一个 prompt 取 24 个贪心 token：

| 变更 | 计划 | 结果 |
|---|---|---|
| 配置 `attn.backend`、`gdn.backend` 设成 `reference` | worker 重载 `layers.attn`、`layers.gdn` | 提交；输出变了（参照 kernel） |
| 设回 `auto` | 同上 | 提交；输出回到原值 |
| 改 `kernels/triton/ops.py`：packed decode 的 scale 乘 0.5（临时的、输出会变的改动） | dry-run 和实际执行都是 worker 重载 `core.plan`、`kernels.triton` | 1.6 秒提交（check+plan 0.97，prepare 0.13，pause 0.02，execute 0.10，commit 0.38）；输出变了 |
| 改回去 | 同上 | 身份回到原值 `6334950edbf50aca`，输出回到原值 |

`nvidia-smi` 的显存前后是 21482 和 21484 MiB。没有发现更新机制的问题。

期间静态检查拦下过一次我自己的错：`kernels/triton/unit.py` 里写了两个模块级的 dict（`manifest.mutable_global`），改成元组后通过。

## 测试

- `uv run pyright` 0 错误；`uv run pytest -n 8` 597 个通过。
- `tests/gpu/`（4090 D，逐个文件跑）全部通过：`test_real_kernels` 22 个，`test_qwen38_partial` 4 个，`test_cuda_graph` 9 个，`test_cuda_graph_adversarial` 3 个，`test_state_rebuild` 1 个，`test_cuda_graph_apply` 4 个，`test_qwen38_overlap` 1 个，`test_qwen38_serve` 2 个。
- `test_real_kernels.py` 新增或改写的：
  - GDN decode 的两个新 kernel 对照参照实现（含一条 padding 行：输出没有定义，状态池不动）；gated RMSNorm 接 strided 的 `z`；
  - 融合的 Q/K norm + RoPE + gate 和 sigmoid·mul 对照参照实现（bf16、fp16、带和不带 gate，decode 和 prefill 的位置）；
  - FlashInfer 的 RMSNorm（bf16、fp32，带和不带 residual）和 SiLU·mul（bf16、fp16）对照参照实现；
  - 上面"review 的修复"表里的五个 worker 栈测试。
- `tests/gpu/conftest.py` 改了：原来只在测试开始前 import 一次 Triton 和 FlashInfer。FlashInfer 的 norm 是 CuTe DSL 编译的，它在第一次调用时才 import 一批子模块，这些子模块在下一个测试里会被 `tests/conftest.py` 的隔离忘掉，再 import 一次就是另一份类对象，编译器不认。现在每个测试结束时记下所有不属于本仓库的模块，下一个测试开始时放回去。

## 还和 SGLang 不一样的地方

这台机器上能做、这次没做的：

- **KV 写入**：这里是两次 `index_put`，SGLang 是一个 `store_kvcache` kernel。它是 SGLang 自己用 C++ 写、由它的 JIT 编译的，不是 Triton，没法照搬源码。每个 attention 层多 1 次 launch（完整模型每步多 16 次）。
- **attention 每步的准备**：每行的 token 槽列表是用 torch 的几个操作拼的，连同拷贝是 13 次 launch；SGLang 用一个 Triton kernel（`create_flashinfer_kv_indices_triton`）。这在 graph 之外、每步一次，不随层数增长。
- **prefill 不走 graph。** SGLang 给 prefill 也录 graph。上面 1 并发 prefill 多出的 8 ms 多半在这里。这是一个新的 plan 种类，不是换 kernel。
- **extend 的 ragged wrapper**：这里的 prefill 用 paged wrapper 在整段序列上算（SGLang 的 `SGLANG_FLASHINFER_USE_PAGED` 那条路）。
- **融合 conv 在 conv 状态和激活 dtype 不同时的退路**（见上）。
- prefill kernel 的启动 warm-up；每步 plan 在 host 上的 0.3 ms；`holder.mem_fraction` 和 `sched.token_budget` 的联动。这三项和第一批文档里写的一样。

留给 SM100 的不变：FlashInfer 的 GDN（SM90 / SM100）、CuTe DSL 的 GDN、`trtllm_mha`。另外 Gemini 的报告里提到，SGLang 在 SM100 加 head dim 256 时默认选的 attention 是 `triton` 而不是 `flashinfer`（两者吞吐差不到 2%）；这里的 `auto` 在有 FlashInfer 时选 FlashInfer。

## 对 SM100 的预期（没有验证）

> 后来 Gemini 在 SM100 上重测了，结果在 [real-kernels-sm100.md](real-kernels-sm100.md)：64 层的 decode 是 SGLang 的 92.7–100.9%，prefill 是 97.7–100.2%。下面是重测之前写的推算。

- Gemini 的归因：64 层时每步比 SGLang 多约 4.07 ms，对应每层约 63 µs 的未融合小算子和 launch 间隙。
- 按这里 4 层的计数换算到 64 层（48 层 GDN + 16 层 attention）：第一批每步比 SGLang 多 2000 多次 launch（每个 GDN 层约 25 次，每个 attention 层约 68 次）。现在 GDN 层不多，每个 attention 层多 1 次，加上每步一次的 6 次，合计约 22 次。
- 所以 decode 应该接近 SGLang。prefill 每步的 launch 也少了一半，但它是 eager，host 一侧的代价在 SM100 上占多少我不知道。两项都要以 SM100 上的重测为准。
