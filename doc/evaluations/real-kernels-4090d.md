# 真 kernel 第一批：Triton GDN 和 FlashInfer attention（2026-11-11，RTX 4090 D，4 层）

M2b（[../specs/stage2-contract.md](../specs/stage2-contract.md) §1.3）在这台机器上能做的部分：把 SGLang `e838360` 的 GDN Triton kernel 和 FlashInfer 的 paged attention 接进 kernel 目录，eager 和 decode graph 都用它们。只在 Qwen3.8-27B 真实结构的前 4 层（3 层 GDN + 1 层 full attention，随机 bf16 权重，checkpoint 7.57 GiB）上验证过；完整模型和只有 SM90 / SM100 才有的 kernel 留给 SM100。

## 结论

1. **接上了，结果对。** `auto` 在这张卡上给 bf16 模型选的是 `triton.gdn.*`（prefill 3 个、decode 2 个）、`triton.norm.rms_gated@1` 和 `flashinfer.attn.paged_extend@1` / `paged_decode@1`，eager 和 graph 是同一套。
   - fp32：FlashInfer 和 Triton 的 chunk kernel 都不收 fp32，`auto` 在 fp32 下留在参照实现上（attention、chunk），其余 Triton kernel 照用。4 层模型 eager、graph 的贪心 token 和 HF 12 步全部相同，和原来一样。
   - bf16：prefill logits 和 HF 的余弦 0.9994–0.9996。贪心 token 和 HF 相同的步数，三个 prompt 分别是 12、10、2（共 12 步），eager 和 graph 相同。换 kernel 之前 eager 是 12、10、12，graph 是 3、1、3。按 §1.3 的口径，bf16 只报告分歧位置。
   - **graph 和 eager 的 token 现在一致。** benchmark 的 12 组配置里 12 组相同，之前是 0 组：之前 graph 用的是另一个 attention kernel（`kernels.torch_graph`）。
2. **这个模型在这张卡上的 decode 已经压在 GEMM 上，再换 kernel 收益很小。** 换完 kernel 以后的 profile（eager，8 行，`torch.profiler`，只算设备上的 kernel）：每步 6.41 ms，其中 GEMM 5.94 ms，其余全部加起来 0.47 ms。
   - LM head 一项 2.63 ms：248320 × 5120 的 bf16 权重 2.5 GB，每步读一遍，是显存带宽；
   - 另外 19 个 GEMM 共 3.31 ms，同样是读权重；
   - 3 个 GDN recurrent kernel 共 0.11 ms，其余每种 kernel 不到 0.05 ms。

   benchmark 里开重叠的 graph 每步 6.0–6.4 ms，已经等于这个设备时间。换 kernel 之前，Triton GDN 让 graph 每步快了 0.3–1.0 ms，FlashInfer 再快 0–0.8 ms（见下表）。只有 4 层时 LM head 占了近一半；完整 64 层时它被摊薄，但每层的 GEMM（0.83 ms）仍是非 GEMM 部分（0.12 ms）的 7 倍。带宽高得多的 SM100 上占比会不同，要在那里重新 profile。
3. **FlashInfer 的意义在于 graph 步的代价不再跟 `batch.max_context_len` 走。** 之前 graph 里的 attention 按表宽算，`max_context` 从 256 调到 4096，8 行的每步时间从 7.39 涨到 8.15 ms；现在两档都是 6.40 ms。
4. **热更新没发现问题。** 在跑着的引擎上改 `kernels.triton`、`kernels.flashinfer` 的代码和 backend 配置，计划、生效、撤回、显存都符合预期（见下）。没有需要写进 `doc/framework-bugs/` 的。
5. **和原生 SGLang 对比（同一天补测，见"和原生 SGLang 对比"一节）：经 HTTP 的吞吐，decode 是它的 96–99%，prefill 是 94%。** 同一个 checkpoint、同一个 tokenizer、同样的请求，SGLang 是 `e838360` 的默认配置。每个 token 的间隔，agxinfer 比它多 0.2–0.3 ms（1 并发 6.00 对 5.80 ms，32 并发 7.55 对 7.35 ms）。这个差距没有逐项归因；SGLang 在这条路径上比我们多的是 GDN decode 的融合 kernel、QK norm + RoPE 的融合和 `sgl_kernel` 的小算子。

## 环境

- RTX 4090 D，24 GB，SM89，驱动 590.44.01；Python 3.12.13，torch 2.11.0+cu130，Triton 3.6.0，flashinfer-python 0.7.2（SGLang `e838360` 钉的版本）。
- `pyproject.toml` 的 `gpu` extra 加了 `triton==3.6.0`（torch 自己带的那个版本）和 `flashinfer_python[cu13]==0.7.2`，`uv.lock` 多了 21 个包（FlashInfer 的依赖，含 `nvidia-cutlass-dsl`）。`cpu` extra 不变，CI 不装它们。
- FlashInfer 第一次给某个 shape / dtype 做 plan 时要编译 kernel 模块，需要 `PATH` 上有 `ninja` 和 CUDA 编译器。`uv run` 会把环境的 `bin` 放进 `PATH`；直接调环境里的 `python` 时要自己加。编译结果放在 `~/.cache/flashinfer`（`FLASHINFER_WORKSPACE_BASE`），Triton 的在 `~/.triton`。

## 做了什么

### 单元

| 单元 | 内容 |
|---|---|
| `kernels.triton`（新） | SGLang 的 Triton kernel，登记成 `triton.<definition>`，对应 SGLang 的 `--linear-attn-backend triton`。6 个 definition：`gdn.conv1d_fn`、`gdn.conv1d_update`、`gdn.gating`、`gdn.chunk_delta_rule`、`gdn.recurrent_update`、`norm.rms_gated`。没有 CUDA 设备的进程里什么都不登记，也不 import Triton |
| `kernels.flashinfer`（新） | FlashInfer 的 paged wrapper，登记成 `flashinfer.attn.paged_extend@1`、`flashinfer.attn.paged_decode@1`，对应 `--attention-backend flashinfer`。同样只在有 CUDA 设备时登记 |
| `kernels.reference` | 加了两个 definition `attn.paged_extend@1`、`attn.paged_decode@1` 和它们的 torch 参照（`paged.py`）；`gdn.conv1d_fn@1` 多了可选的 `seq_lens_cpu`（SGLang 的 kernel 靠它定 launch grid）；decode 的两个 GDN 参照把负槽号当 padding；`gdn.gating@1`、`norm.rms_gated@1` 的容差从逐位改成有容差 |
| `layers.attn` | 两族 kernel 各有一个 backend：原来的 `TorchNativeAttnBackend`，和新的 `PagedAttnBackend`（`paged.py`，SGLang `FlashInferAttnBackend` 的普通路径）。`resolve` 按阶段选一族 |
| `layers.gdn` | graph 桶里的 padding 行槽号用 -1；把模型是不是 fp32 放进 shape；把 host 上的序列长度交给 prefill conv |
| `core.plan` | capture 后的自检只对 1 行的 graph 做 |
| `layers/solutions.py` | `resolve` 多了 `shared`：不属于 backend 自己的 definition（KV 写入） |

`kernels.torch_graph` 留着：fp32 的 graph、没有 FlashInfer 的 GPU，以及 CPU 上用替身 recorder 跑回放路径的测试还用它。bf16 加 FlashInfer 时不会选到它。

### 搬了哪些文件

全部来自 SGLang `e838360` 的 `python/sglang/kernels/ops/`，放在 `engine/kernels/triton/`，文件头注明出处，约 4700 行：

| 这里 | SGLang |
|---|---|
| `causal_conv1d.py` | `mamba/causal_conv1d_triton.py` |
| `fused_gdn_gating.py`、`fused_sigmoid_gating_recurrent.py` | `attention/linear/` 下的同名文件 |
| `fla/chunk.py`、`chunk_delta_h.py`、`chunk_fwd.py`、`chunk_o.py`、`cumsum.py`、`index.py`、`l2norm.py`、`op.py`、`wy_fast.py`、`utils.py`、`layernorm_gated.py` | `attention/fla/` 下的同名文件 |

- **改动只有去掉对 `sglang.srt` 的依赖。** kernel 本体没动。去掉的是 AMD / Intel / NPU / CPU 的分派，以及几个环境变量开关（固定成默认值：引擎的身份由快照内容和配置决定，环境变量是没被记录的输入）。`fla/utils.py` 只留这些 kernel 用到的部分，`fla/layernorm_gated.py` 只留前向 kernel 和它的 launcher。
- **`ops.py` 是适配层。** definition 的签名是参照实现的，SGLang kernel 的参数顺序和它不一样；每个函数照 `GDNAttnBackend` / `TritonGDNKernel` 的调法调一次。
- **搬源码而不是依赖 `sglang` 包**：单元要在快照里自足，agent 也才能热改 kernel。
- 这些文件第一行是 `# type: ignore`：上游代码没有类型标注，pyright 在里面有一百多个错。另外，CI 的环境没有装 Triton 和 FlashInfer，pyright 会把这两个单元目录里的 `import triton`、`import flashinfer` 解析成同名的单元目录本身（运行时不会）；自己写的几个文件因此按名字取库（`importlib`）。

### attention 为什么要新的 definition

FlashInfer 把一步拆成两半：在 graph 外按这一步的长度做 plan（在 host 上），在 graph 里只跑 kernel。`attn.decode@1` 那种"一个函数读 `req_to_token`"的签名装不下它。所以：

- **新 definition 的 solution 不是函数，是造 operator 的东西**（`kernels/reference/paged.py` 的文档写了约定）：
  - 构造：头数、head dim、dtype、设备，以及 graph 桶的固定 buffer（eager 用时没有）；
  - `plan(qo_indptr_cpu, kv_indptr_cpu, kv_indptr, kv_indices)`：每步在 graph 外调一次。每行的 token 槽排成一段（page size 按 1 个 token 算，和 SGLang 一样，所以 KV 池的布局不用改），长度在 host 上给出，plan 不用等设备；
  - `run(q, k_cache, v_cache, scaling)`：eager 或者在 capture 里跑。
- **operator 在第一次 `forward` 时造**，因为头数到那时才知道；那次 `forward` 也负责 plan。graph 桶的第一次 `forward` 是 capture 前那次 eager warm-up，所以被录下来的 `forward` 里只有 `run`，之后每次回放都在 `prepare` 里 plan。
- decode wrapper 的第一次 plan 用 FlashInfer 自己的（它会编译或加载 kernel 模块），之后用 `fast_decode_plan`，和 SGLang 给 graph 的 wrapper 装的一样。
- **extend 现在走的是 SGLang 不用 ragged wrapper 的那条路**（它的 `SGLANG_FLASHINFER_USE_PAGED`）：先把新 token 的 K/V 写进池，再让 paged prefill wrapper 在整段序列上做 causal attention。SGLang 默认是 ragged + paged 再合并，没有搬。
- kernel 目录重新登记以后（kernel 单元重载），operator 在下一步重新造，不会接着跑旧代码的。

### 每个阶段选哪一族

`resolve` 在 worker 启动时定，和 SGLang 的 `HybridAttnBackend`（prefill 一个 backend、decode 一个）是一个意思：

- `auto`：有库提供 paged kernel（FlashInfer）就用 paged，否则用原来的；这是 SGLang 在 SM89 上的默认。
- 写了名字：这个名字有原来那一族的 solution 就用原来的（`reference`），否则用 paged（`flashinfer`）；两族都没有就启动失败。
- graph 跟 decode 走同一族。

### 三处行为变化

1. **graph 桶里 GDN 的 padding 行，槽号从 0 改成 -1**，和 SGLang 的 `pad_slot_id` 一样。
   - 原因：原来所有 padding 行都指向槽 0。参照实现是先全部读出、算完再写回，结果确定；Triton kernel 每行一组 program 并行读写同一个槽，结果取决于谁先写，graph 的自检第一次就挂了（回放和 eager 的 logits 差 0.33）。
   - -1 的含义：kernel 不为这一行读写状态，这一行的输出没有定义（Triton 的 decode conv 直接跳过，输出是没初始化的内存）。真实行不受影响。还省掉了 padding 行每层 3 MB 状态的读写。
   - 参照实现遇到负槽号时仍在槽 0 上算，结果确定。
2. **capture 后"回放等于 eager"的自检只对 1 行的 graph 做。**
   - 原因：padding 行的输出没有定义以后，多行的桶里没有一行可以拿来比。试过只比第一行（让它用 padding 行和 scratch 槽）：padding 行和它共用 KV 的 padding 槽，会把没定义的 K/V 写进去，第一行读到的也不确定。
   - 1 行的桶没有 padding 行，结果有定义。配置里没有 1 这个桶时，专门录一个 1 行的 graph 做自检，录完就关。
   - 代价：只在某个 batch 大小下才出错的 capture 问题，启动时发现不了，要靠 `tests/gpu/`（那里用真请求比 graph 和 eager）。
3. **显式写的 attention backend 不再要求自带 KV 写入。** 原来"显式指定的 backend 必须覆盖该阶段的每个 definition"，而 `kv.store@1` 也算在内，于是 `attn.backend = flashinfer` 启动就失败（新测试抓到的）。SGLang 里写 KV 是池的 `set_kv_buffer`，和 attention backend 无关。现在这类 definition 标成 `shared`：点名的 backend 没有，就按 `auto` 选。

## 数字

`tests/gpu/bench_cuda_graph.py --max-context 256 4096 --repeats 2`，其余默认（prompt 32，预热 16 步，测 128 步，`batch.max_rows=8`，KV page size 4）。每步毫秒，中位数。depth 2 是开一步重叠。

`max_context` 4096：

| batch | depth | 方式 | 参照 kernel | + Triton GDN | + FlashInfer attention | 倍率 |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1 | eager | 9.38 | 7.61 | 7.61 | 1.23× |
| 1 | 1 | graph | 6.84 | 6.58 | 6.80 | 1.01× |
| 1 | 2 | eager | 6.85 | 6.44 | 6.13 | 1.12× |
| 1 | 2 | graph | 6.30 | 6.02 | 6.00 | 1.05× |
| 4 | 1 | eager | 10.35 | 8.55 | 8.15 | 1.27× |
| 4 | 1 | graph | 7.92 | 7.31 | 7.18 | 1.10× |
| 4 | 2 | eager | 8.00 | 7.29 | 6.46 | 1.24× |
| 4 | 2 | graph | 7.27 | 6.66 | 6.32 | 1.15× |
| 8 | 1 | eager | 11.20 | 9.48 | 8.19 | 1.37× |
| 8 | 1 | graph | 8.91 | 7.95 | 7.41 | 1.20× |
| 8 | 2 | eager | 9.07 | 8.04 | 6.55 | 1.39× |
| 8 | 2 | graph | 8.15 | 7.19 | 6.40 | 1.27× |

- `max_context` 256 时最后一列的 graph、depth 2 是 6.01 / 6.32 / 6.40 ms，和 4096 一样。
- 8 行、graph、depth 2 的吞吐从 982 涨到 1250 token/s。
- graph 步在 host 上的提交时间从 0.79 ms 涨到 1.08 ms：多出来的是每步的 FlashInfer plan 和几次小的 host 到设备拷贝。开重叠时它被上一步的设备时间盖住；不开时就是每步多 0.3 ms，1 行不开重叠的 graph 因此没有变快。
- eager 的提交时间从 8.96 ms 降到 5.26 ms（8 行）。
- 冷缓存下第一次编译：FlashInfer 的 prefill 模块约 18 秒（bf16、GQA 每组 6 个头时 decode 走 tensor core，用的也是它；不走 tensor core 的 decode 模块约 10 秒）；Triton 的 chunk kernel 连 autotune 约 10 秒，其余每个约 1 秒。之后从磁盘缓存加载。**Triton 的 prefill kernel 没有 warm-up，冷缓存时第一个请求会等这次编译。**
- 录 4 个桶的 graph 共 0.1–0.2 秒。第一次录时 `kept_bytes` 是 532 MiB，里面有 FlashInfer 的 384 MiB workspace（它在第一个 operator 造出来时分配，正好落在统计窗口里）；重载后再录是 60 MiB。workspace 不是 graph 的显存，这个统计把它算进了 `plan.graph_memory_bytes` 的预算。

## 和原生 SGLang 对比

用户允许在这台机器上装原生 SGLang 以后补测的。

### 怎么测的

- **SGLang**：`e838360` 的源码（`~/sglang-e838360/python`，不安装，放在 `PYTHONPATH` 上），依赖照它的 `pyproject.toml` 装进单独的环境 `~/sglang-e838360/venv`（torch 2.14.1+cu130，flashinfer 0.7.2，sglang-kernel 0.4.10）。启动：

  ```sh
  python -m sglang.launch_server --model-path <checkpoint> --dtype bfloat16 \
      --attention-backend flashinfer --context-length 4096 --max-running-requests 32
  ```

  其余默认：重叠调度开，radix cache 开，`chunked_prefill_size=4096`，`max_prefill_tokens=16384`，`mem_fraction_static` 自动定成 0.705，decode graph 录 1、2、4、8、12、16、24、32 行，另外还录 prefill graph。它选的 kernel：FlashInfer attention，Triton GDN（`packed_decode=True`，并用上了投影解包 + conv 的融合 kernel）。
- **让 SGLang 加载这个 checkpoint 要两处调整**：
  - `Qwen3_5ForConditionalGeneration` 是多模态结构，SGLang 会去找图像处理器的文件。在 config.json 里加 `"language_model_only": true`（SGLang 认 checkpoint 自己的这个声明）；`--language-model-only` 这个命令行开关在 `e838360` 不支持这个结构。
  - tokenizer 用 HF 缓存里 `Qwen/Qwen3.6-27B` 的（本机已有，词表 248044）。测试里的词级 tokenizer 不行：SGLang 会"修正"它的组件，结果 prompt 变成 0 个 token，启动预热就挂。
- **agxinfer**：`deploy/qwen38.py serve --cuda-graph auto --overlap --max-rows 32 --max-context 4096 --graph-buckets 1 2 4 8 16 32`，graph 录到 32 行是为了和 SGLang 对齐。prefill 那一组再加 `--token-budget 4096`：SGLang 一步 prefill 的上限是 `chunked_prefill_size`，这张卡上是 4096（见"prefill 每步多少 token"）。decode 那一组是带着 `--token-budget 16384 --mem-fraction 0.75` 测的，prompt 只有 32 个 token，这两项不影响结果（不带它们时是 164.9 / 627.3 / 1230.8 / 2334.5 / 4021.9 token/s）。
- **负载**：`tests/gpu/bench_serving.py`（新）。每档并发有那么多个客户端，各自一个接一个地发 `/v1/completions`（流式、贪心、`ignore_eos`）。每个请求的 prompt 全程唯一。
  - decode：prompt 32 个 token，生成 256 个；每个客户端 4 个请求。
  - prefill：prompt 2000 个 token，生成 8 个；每个客户端 8 个请求。
- 两边在同一个 prompt 上生成的前 4 个 token 相同，之后分开（bf16、随机权重）。

### 结果

decode（输出 token/s；每 token 间隔是每个请求第一个 token 之后的平均间隔，取中位数）：

| 并发 | SGLang token/s | agxinfer token/s | 比例 | SGLang 每 token | agxinfer 每 token |
|---:|---:|---:|---:|---:|---:|
| 1 | 171.0 | 164.8 | 96.4% | 5.80 ms | 6.00 ms |
| 4 | 656.7 | 627.1 | 95.5% | 6.01 ms | 6.31 ms |
| 8 | 1276.8 | 1226.4 | 96.1% | 6.15 ms | 6.40 ms |
| 16 | 2366.2 | 2340.1 | 98.9% | 6.44 ms | 6.66 ms |
| 32 | 4164.3 | 4036.9 | 96.9% | 7.35 ms | 7.55 ms |

prefill（输入 token/s；首 token 时间取中位数）：

| 并发 | SGLang token/s | agxinfer token/s | 比例 | SGLang 首 token | agxinfer 首 token |
|---:|---:|---:|---:|---:|---:|
| 1 | 19166 | 17926 | 93.5% | 63 ms | 65 ms |
| 4 | 30801 | 29120 | 94.5% | 162 ms | 181 ms |
| 8 | 35506 | 33325 | 93.9% | 258 ms | 289 ms |

- SGLang 重复跑一次是 19992 / 31685 / 36098 token/s，首 token 59 / 159 / 251 ms：两次之间差 4% 以内。
- decode 的首 token 时间：SGLang 17–81 ms，agxinfer 24–93 ms。

### prefill 每步多少 token

两边的调度策略是同一个：先来先服务，能组 prefill 就先跑 prefill，不混合，按一步的 token 预算装请求，最后一个装不下的请求切块（`policies.fcfs`、`policies.prefill_first` 就是照 SGLang 和 mini-sglang 搬的）。不同的只是预算的值：

| | 每步 prefill 的上限 | 这张卡上的值 |
|---|---|---|
| SGLang | `chunked_prefill_size`，没指定时按显存大小定 | 4096 |
| agxinfer | `sched.token_budget`，固定默认值 | 2048 |

SGLang 的 `max_prefill_tokens`（16384）是另一个上限，开着分块时先碰到的是 4096：它的日志里一步最多 `#new-token: 4096`。

同一个 prefill 负载、8 并发，agxinfer 在三种预算下（吞吐都是 3.3 万 token/s 左右，受 GEMM 限制）：

| `--token-budget` | 首 token 中位数 | 说明 |
|---:|---:|---|
| 2048（默认） | 259 ms | |
| 4096 | 289 ms | 和 SGLang 的每步上限相同；SGLang 是 251–258 ms |
| 16384 | 427 ms | 8 个请求全放进一步，一起出结果；要 `--mem-fraction 0.75`，否则 OOM |

**我第一次对比时把 `--token-budget` 设成了 16384，以为这是在对齐 SGLang 的 `max_prefill_tokens`，其实是它实际每步上限的 4 倍。** 当时首 token 427 对 258 ms 的差距和下面的 OOM 都是这个设置造成的，不是调度策略不同。
- **agxinfer 的默认配置**（graph 桶到 8 行，`--token-budget 2048`，`--mem-fraction 0.85`）下 decode 是 164.8 / 624.0 / 1225.8 / 2199.2 / 3947.7 token/s：16 并发是 SGLang 的 93%，32 并发是 95%。超过 8 行的步走 eager，开重叠时差得不多。

### 测的过程里碰到的

- **第一版脚本不同并发档位之间有重复的 prompt，SGLang 的前缀缓存命中了**（它的日志里 `#cached-token: 3968`），8 并发的 prefill 看起来是我们的 1.7 倍。改成全程唯一以后是上表的 94%。agxinfer 现在没有前缀复用（M2d），真实负载里有重复前缀时这个差距会回来。
- **`--token-budget 16384` 加默认的 `--mem-fraction 0.85` 会让 worker OOM**，引擎停止服务，只能重启。原因是这两个配置之间没有联系，而不是 prefill 本身费显存：
  - 一步 prefill 的瞬时显存和 token 数成正比，进程内量到每个 token 约 200 KiB（峰值已分配量）：2000 个 token 395 MiB，4000 个 781 MiB，16000 个 3121 MiB。PyTorch 的分配器为这个峰值保留的更多，16000 个 token 那一步之后保留量涨了 5.7 GiB。
  - 默认的 0.85 分给状态池 12.41 GiB，权重 7.6 GiB，graph 和 FlashInfer 的 workspace 约 0.6 GiB，剩下不到 3 GiB。4096 个 token 的步放得下（实测，没有 OOM），16000 个的放不下。
  - SGLang 是从每步上限反推池的大小：留 `512 MB + 1.5 MB × chunked_prefill_size`，再加 graph 的份，剩下的才给权重和池。这张卡上是 0.705，状态池 8.65 GB（KV 4.56，GDN 4.09），建完池、录完 graph 以后还空着 5.3 GB。按同一个公式，每步 2048 个 token 对应的是 0.83 左右，和我们的 0.85 接近，所以我们的两个默认值彼此大致一致；只是把 `sched.token_budget` 调大时，`holder.mem_fraction` 不会跟着降。
  - **同样大小的步，我们用的显存和 SGLang 是一个量级。** 都是每步最多 4096 个 token、同一个 8 并发 prefill 负载，`nvidia-smi` 的占用：agxinfer 从 21532 涨到 23062 MiB（+1.5 GiB），SGLang 从 18692 涨到 19430 MiB（+0.7 GiB），另外它启动时已经给 prefill graph 留了 0.9 GB。空闲时我们多占的 2.8 GiB 是状态池更大（KV 171 万 token 对 119 万，GDN 槽 656 对 455）。
- SGLang 冷缓存的第一次启动约 95 秒，其中录 prefill graph 56–64 秒（50 个长度档位，含 Triton 的编译和 autotune）；缓存热了以后录 prefill graph 4.3 秒。

## 在跑着的引擎上热改

`deploy/qwen38.py serve`（4 层、bf16、`--cuda-graph auto`、`--max-rows 8`），前端发 completions 请求，每次变更前后用同一个 prompt 取 24 个贪心 token：

| 变更 | 计划 | 结果 |
|---|---|---|
| 改 `kernels/triton/ops.py`：decode conv 不带激活（临时的、输出会变的改动） | worker 重载 `core.plan`、`kernels.triton` | 2.3 秒提交（check+plan 0.92，prepare 0.29，pause 0.02，execute 0.39，commit 0.37）；输出变了，graph 重录 |
| 改回去 | 同上 | 身份回到原值 `e1fccc5cc57ac2a7`，输出回到原值 |
| 改 `kernels/flashinfer/ops.py`，连续 3 次，再改回 | worker 重载 `core.plan`、`kernels.flashinfer` | 4 次都提交；输出每次都是原值；`nvidia-smi` 的显存一直是 21488 MiB |
| 配置 `attn.backend`、`gdn.backend` 都设成 `reference` | worker 重载 `layers.attn`、`layers.gdn` | 全部换成参照 kernel；`auto` 下不再录 graph，原因写在日志里（`attn.decode@1` 不可捕获） |
| 设成 `flashinfer`、`triton` | 同上 | 换回真 kernel，graph 重录，输出回到原值 |
| 设回 `auto` | 同上 | 输出是原值，显存 21490 MiB |

参照 kernel 和真 kernel 在这个 prompt 上给出的 24 个 token 不同。bf16、随机权重下候选很接近，这在口径之内，没有单独量 logits 的差。

## 测试

- `tests/gpu/test_real_kernels.py`（新，11 个，23 秒）：
  - 每个 Triton solution 对照参照实现，在同样的输入和状态池上，容差用 definition 的；包括一条 padding 行；
  - FlashInfer 的 operator 对照参照 operator：eager，以及 graph 桶的固定 buffer（先正常 plan，再换长度用 fast plan）；
  - worker 栈（4 层、bf16）：`auto` 选的全是真 kernel；3 行进 4 行的桶、换桶、换请求，graph 和 eager 的 logits、token、状态一致；
  - 两个 kernel 单元带着在途请求重载两次，接着生成的 token 和没重载的 worker 相同；
  - 按 SGLang 的名字点名能启动；fp32 点名 FlashInfer 启动失败。
- `tests/units/executors/test_paged_attention.py`（新，CPU，进 CI）：CI 里没有库提供 paged kernel，测试把 backend 切到 paged 一族的参照 operator 上，整条 worker 栈生成的 token 和 logits 与原来那一族相同（整段、分块、mixed、decode）。
- `tests/units/kernels/`：paged 参照和原来的参照逐位相同。
- `tests/gpu/conftest.py`（新）：GPU 测试进程在任何测试之前先 import 一次 Triton 和 FlashInfer。`tests/conftest.py` 的隔离会在每个测试后忘掉它 import 的模块，而 FlashInfer 依赖的 CuTe DSL 是原生扩展，同一进程里第二次 import 会直接 abort。引擎进程不受影响：重载只丢单元自己的模块。
- 原有的 `tests/gpu/` 全部通过：`test_qwen38_partial`、`test_cuda_graph`、`test_cuda_graph_adversarial`、`test_cuda_graph_apply`、`test_state_rebuild`、`test_qwen38_overlap`、`test_qwen38_serve`（bf16）。`test_qwen38_serve` 用的是测试里的词级 tokenizer，生成的 token 解码出来是空串，只说明链路通、开不开重叠一致。
- `uv run pyright` 0 错误；`uv run pytest -n 8` 596 个通过。

## 没做的

留给 SM100（这张卡是 SM89，跑不了）：
- FlashInfer 的 GDN decode / prefill（SM90、SM100；SM100 上 decode 要求 ssm state 是 bf16），CuTe DSL 的 GDN，`trtllm_mha`；
- 完整模型上和原生 SGLang 的对比（`tests/gpu/bench_serving.py` 两边通用），以及那里的 profile。

这台机器上还能做、这次没做的：
- **融合路径**：QK norm + RoPE + gate、`fused_sigmoid_mul`、GDN decode 的 packed decode 和投影解包 + conv、两个输入投影合成一次 GEMM。都要加 definition 并改 `models.qwen38` 的前向。按上面的 profile，除了合并 GEMM，其余在这张卡上每层最多省 0.1 ms 量级。
- **小算子**：norm、silu、RoPE、采样（`sgl_kernel` / FlashInfer）。profile 里它们每个不到 0.05 ms。
- extend 的 ragged wrapper。
- prefill kernel 的启动 warm-up。
- 默认的 graph 桶只到 8 行（`core/plan/graph.py` 的 `DEFAULT_MAX_BUCKET`），理由是当时 graph 里的 attention 按 `max_context` 算。换成 FlashInfer 以后这个理由不成立了，更大的桶没有重新量。
- 每步 plan 在 host 上多出的 0.3 ms。
- `holder.mem_fraction` 没指定时按 `sched.token_budget` 来定（SGLang 的做法），以及按显存大小定默认的预算；现在把预算调大而不降 `--mem-fraction` 会 OOM。
