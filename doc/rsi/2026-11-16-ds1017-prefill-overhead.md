# DS 1017 的 prefill：开销在哪、稳态吞吐，以及一次显存不够带崩引擎（2026-11-16 下午）

- 记录人：Devin。当天 14:30 接手 Antigravity 的会话。14:15 以前的 profiler 结果是 Antigravity 量的，下面标了出处；其余是接手以后量的。
- 对应：[../specs/v1-ds1017.md](../specs/v1-ds1017.md) 第 10.8、10.9 节；PR #66（分支 `dev/ds1017-ep`）。接着 [2026-11-16-ds1017-multi-gpu-hot-apply-and-fusion.md](2026-11-16-ds1017-multi-gpu-hot-apply-and-fusion.md) 末节「prefill 重测」往下做。
- 机器：SM100 节点 `n1`，GPU 4–7 跑 4 卡全量模型的引擎（`--ds-backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 64 --disable-radix-cache`，`--mem-fraction` 默认 0.85），GPU 1、2 跑 GPU 单测；`dev-server` 跑 pyright 和 CPU 单测。

## 结论

1. **和 DeepInfer 的 benchmark 同样设置时，prefill 经 HTTP 的稳态吞吐是它的 96.5%（不开重叠）和 98.0%（开重叠）**：每卡 43,393 和 44,069 tok/s，DeepInfer 是 44,952。同样设置指每步 32,768 个 token、门控强制随机（它的 `examples/benchmark.py` 写死 `DEEPINFER_FORCE_LOAD_BALANCE=1`）。
2. **上午「只有四分之一」的大头在主机上**：`Layers.lay_out()` 在 CPU 上算一步的索引张量，32,768 个 token 的一步里它比前向本身还长。搬到设备上以后，一步从约 2.4–2.6 秒降到 0.72–0.75 秒（强制随机）。
3. **用模型自己的路由是每卡 40,170 tok/s**，是 DeepInfer 那个数的 89.4%，但口径不同：DeepInfer 没有不强制随机的数。一步多出约 75 ms；Antigravity 的 profile 里多在 MoE（各 rank 的专家负载不均），这个分解我没有复测。
4. **每步 32,768 个 token 配默认的 `--mem-fraction 0.85`，显存贴着上限**：压测里一个 rank 的一步 OOM，另外三个 worker 在融合 MoE 的屏障里等到超时后进程退出，引擎只能重启。每步 16,384 个 token 时每张卡至少空 11.45 GiB，门控同样强制随机时是 DeepInfer 的 91.6%（不开重叠）和 93.4%（开重叠）。
5. 这半天热改提交 13 次，冷启动 1 次（就是上面那次崩溃以后）。

## 改了什么

| 单元 | 改动 | 谁 |
|---|---|---|
| `layers.ds1017` | `lay_out()` 只把计划里每行的几个数送上设备，每个 token 的索引张量（窗口、压缩器的条目、n-gram）在设备上算 | Antigravity |
| `layers.ds1017` | `lay_out()` 里留在窗口里的 token 下标（`keep`）按每行的 token 数直接算，不用 `torch.nonzero`。`nonzero` 要等设备算完才知道结果多长，开重叠时等的是还在跑的上一步 | Devin |
| `executors.pytorch` | 计划里的输入 token 从列的字节直接成张量（`torch.frombuffer`），不再一个 token 一个 Python int。32,768 个 token：2.58 ms 降到 0.008 ms（n1 的 CPU 上单独量的）。`check_plan` 拒绝不是 `i64` 的 token 列 | Devin |
| `kernels.deepinfer` | 配置键 `deepinfer.force_random_gate`（启动时绑定，默认 false）：融合门控给每个 token 随机挑专家，各 rank 负载均匀，和 DeepInfer 的 benchmark 一样。只为对照用，打开以后生成的文本没有意义 | Antigravity |

没有改的：Antigravity 的计划里有一条「`_warm_prefill()` 预热 `topk = 512`」。它的依据是一次 4 秒的停顿，但那次停顿之前已经跑过很多 `topk = 512` 的步，原因并没有确认，所以没改。

## 怎么量的

- 稳态：256 个客户端闭环，每个客户端发一条约 3,880 个 token 的提示词、只要 1 个输出 token，收到回答马上发下一条。每张卡常有 64 条请求，所以每一步都装满预算。先跑 10 秒，再量 40 秒。
- 引擎里不放探针。4 个调度域一起走步，回答成批到达，一批就是一步；吞吐按第一批到最后一批之间的 token 数和时间算，步的周期取相邻两批的间隔。
- 提示词：随机 token id 解码成的文本（3,860–3,884 个 token）；另一种是 50 个常用英文词随机排的（3,801 个 token）。同一轮里不重复。
- 每步的 token 数用 `sched.token_budget`，门控用 `deepinfer.force_random_gate`，重叠用 `sched.overlap`，都是 `apply --config` 热改的。
- DeepInfer 的对照是任务 `20261115T120455Z-a5332691` 的 `bench-short`：进程内只计 `llm_engine.step()`，9 步 294,912 个 token 用 6.561 秒，一步 729.0 ms，每卡 44,952 tok/s。它每条请求 4,096 个 token、一步正好 8 条；我们的 `batch.max_context_len` 是 4096，提示词要留一个输出 token，所以是 3,880 左右、一步 8 条多一点。
- 脚本在 n1 的 `/tmp/devin-ab/`（`bench_prefill.py`、`ab_prompts.py`、`mem_probe.py`），没有进仓库。

## 开销在哪

### 数是怎么一步步上来的（每卡，约 3.9K token 的提示词）

| 时间 | 每卡 tok/s | 一步 32,768 token | 当时的情况 |
|---|---:|---:|---|
| 上午 | 2,666 | 每步只有 1 条请求 | `sched.token_budget = 4096`，测量窗里还混着 JIT 编译 |
| 中午 | 10,941–11,459 | 约 2.6 s | 预算改成 32,768 并预热；`lay_out()` 在 CPU 上 |
| 13:52–14:14（Antigravity，带探针，64 条一轮） | 32,646–34,989（真实路由）、36,355–37,287（强制随机） | 720–747 ms（强制随机）、812–876 ms（真实路由） | `lay_out()` 搬到设备上；主机一步多数约 90 ms（86–163 ms） |
| 14:50（稳态，不带探针） | 43,393 / 44,069（强制随机，不开 / 开重叠） | 755.3 / 743.7 ms | 见下一节 |

- `lay_out()` 在 CPU 上时，Antigravity 的 profiler 里 CPU 自身时间最多的五个算子是 `aten::add` 266 ms、`aten::where` 248 ms、`aten::repeat_interleave` 183 ms、`aten::sub` 148 ms、`aten::ge` 133 ms，都是 `[32768, 128]` 的窗口索引运算。它记录的整步是主机 1,710 ms、整步 2,432 ms（这两个数我只在它的小结里看到，没找到原始日志行）。
- 64 条请求一轮的量法比稳态低，是量法的原因：每张卡一轮只有 16 条、不到两步。头一步只装进先到的一两条请求，最后一步不满，4 个调度域分到的请求数还不一样（18 条和 14 条），整轮要等最慢的那张卡。稳态里这些都摊掉了。

### 现在一步里的时间（强制随机，一步 32,768 个 token）

Antigravity 13:57 的 Kineto trace，按发出 kernel 的引擎源码行归并，kernel 时间合计 709.2 ms（1,981 个 kernel）。这是在我的两处小改之前量的，那两处不动 kernel。DeepInfer 的数取自它自己的 `stats_logger`（`bench-short.log`，prefill 的第 9–19 步），两边的分段不完全对应，只能大致比。

| 部分 | 本引擎 | DeepInfer |
|---|---:|---:|
| MoE（`bf16_mega_gate` + `fp8_fp4_mega_moe`） | 264.2 ms | `mlp` 260–268 ms（另有一个窗口 298 ms） |
| 稀疏 attention（`sparse_attn_out`） | 152.9 ms | `prefill_core_attn` 130–133 ms |
| attention 的输出投影（`_wo`） | 72.8 ms | |
| 子层之间的一步（`hc_step`，`mega_mhc`） | 70.9 ms | |
| attention 的输入投影（`attn_qkv`） | 61.0 ms | |
| Engram 取表（`engram_fetch`） | 35.0 ms | |
| indexer（`index_topk`、top-k、query 量化） | 15.1 ms | `prefill_index_attn` 8.9–9.0 ms |
| attention 里拼源和取下标 | 10.8 ms | |
| Engram（`engram`） | 7.2 ms | |
| 其余（`lay_out` 0.45 ms、装载 latent 0.38 ms、压缩器等） | 约 19 ms | |
| 整步 | 720–747 ms（探针量的墙钟） | `device_forward` 719–725 ms |

- 一步本身和 DeepInfer 差不多。稳态里一步的周期是 755.3 ms（不开重叠）和 743.7 ms（开重叠），DeepInfer 是 729.0 ms。
- 不开重叠时多出的约 12 ms 是两步之间的空档：结果回到调度器、下一步的计划发下来（Antigravity 的探针里是 7–10 ms）。开重叠以后下一步在这一步跑着的时候就排好了。
- 开重叠以后还差的约 15 ms（2%）没有再细分。一步开头主机要先改 batch 的表、排 `Layout`、把 token 送上设备，这段时间设备是空的（Antigravity 的 trace 里一步之内设备空了 18.7 ms，大多在开头的 16 ms 里）。

## 稳态吞吐（4×SM100，经 HTTP，每卡）

| 每步 token | 门控 | 重叠 | 提示词 | tok/s | 步的周期（中位数，10%–90%） | 相对 DeepInfer |
|---:|---|---|---|---:|---:|---:|
| 32,768 | 强制随机 | 关 | 随机 token | **43,393** | 755.3 ms（742.9–766.9） | **96.5%** |
| 32,768 | 强制随机 | 开 | 随机 token | **44,069** | 743.7 ms（725.4–760.0） | **98.0%** |
| 32,768 | 模型自己的路由 | 开 | 随机 token | 40,170 | 818.9 ms（795.3–839.7） | 89.4% |
| 32,768 | 模型自己的路由 | 开 | 50 个词 | 引擎崩溃，见下一节 | | |
| 16,384 | 强制随机 | 关 | 随机 token | 41,198 | 396.3 ms（389.7–406.8） | 91.6% |
| 16,384 | 强制随机 | 开 | 随机 token | 42,007 | 390.1 ms（381.0–400.5） | 93.4% |
| 16,384 | 模型自己的路由 | 开 | 随机 token | 39,207 | 417.4 ms（407.8–427.0） | 87.2% |
| 16,384 | 模型自己的路由 | 开 | 50 个词 | 38,983 | 419.6 ms（409.2–430.3） | 86.7% |
| 16,384 | 模型自己的路由 | 关 | 随机 token | 38,534 | 425.7 ms（416.8–433.2） | 85.7% |

- 每一行 40 秒里没有一条请求出错（崩溃的那一行除外）。预热以后没有再碰到几秒的 JIT 停顿：步的周期最长 892 ms（32,768）和 456 ms（16,384）。
- 模型自己的路由那几行和 DeepInfer 的数不是同口径。要比，得让 DeepInfer 也不强制随机再跑一次；它的 benchmark 把这个开关写死了，引擎占着 4 张卡时也腾不出 4 张连续的卡，这次没做。
- 每步 16,384 比 32,768 每个 token 慢约 5%（23.8 µs 对 22.7 µs，强制随机、开重叠）：每步固定的开销摊在一半的 token 上。
- 32,768、模型自己的路由、不开重叠的稳态没有量到：轮到它之前引擎崩了，之后没有在 0.85 下再跑 32,768 的真实路由。
- EPLB（带冗余槽位）以后真实路由的 prefill 没量：改槽位数要重读权重，约 20 分钟一次。

## 显存不够带崩引擎

### 经过

- 14:54，每步 32,768 个 token、模型自己的路由、开重叠，提示词刚从随机 token 换成 50 个词，回答了 133 条请求以后，rank 3 的第 2258 步 OOM：要 2.00 GiB，卡上空 1.93 GiB，PyTorch 已分配 160.46 GiB，另有 5.24 GiB 留在缓存里但是碎的，进程一共占 176.23 GiB（卡 178.35 GiB）。
- 另外三个 rank 这一步卡在融合 MoE kernel 的屏障里等 rank 3，60 秒后是设备断言（`deep_gemm/comm/barrier.cuh:21 Timeout`），接着 `CUDA error: unspecified launch failure`，进程 `terminate` 退出。rank 3 的进程还活着，空转。
- 之后的请求都回 400。我接着发的 `apply --config '{"sched.overlap": false}'`（变更 24）等不到三个 worker 对 Pause 和 Abort 的回答，结果是 `failed`，诊断里有 `change.abort_unacknowledged` 和 `change.recovery_required`。进程已经没了，这是规格里写明的结果，不是更新机制的 bug，所以没有写 framework-bugs 报告。
- 15:03 `agxinfer stop`（干净返回，四张卡的显存都放了），崩掉的运行目录改名留在 n1 的 `/workspace/agxinfer-env/run-ep-crashed-20261116-1456`，再用同样的参数 `serve`，15:23 起来，20.5 分钟。

一个 rank 出错、其他 rank 跟着退出，是 10.9 节里已经列着的「没有做组内恢复」。

### 显存是怎么用的

重启以后在引擎上放探针量的（另一份工作区里给 executor 的 `submit` 加几行，每个大的 prefill 步打印 `torch.cuda.memory_allocated()`、这一步里的峰值、`memory_reserved()` 和卡上的空闲；量完撤掉）。不开重叠，每组 25 秒。

| 每步 token | 门控 | 步开始前已分配 | 一步里的峰值 | 一步多用 | 卡上最少空闲 | 分配失败后重试 |
|---:|---|---:|---:|---:|---:|---|
| 32,768 | 强制随机 | 155.9 GiB | 163.5 GiB | 7.6 GiB（头一步 8.3） | 3.4–5.2 GiB | 33 步里 3 个 rank 各 1 次 |
| 16,384 | 模型自己的路由，随机 token | 155.2–155.6 GiB | 159.3 GiB | 4.2 GiB | 11.45 GiB | 没有 |
| 16,384 | 模型自己的路由，50 个词 | 155.2–155.6 GiB | 159.3 GiB | 4.2 GiB | 11.45 GiB | 没有 |

- 一步多用的显存和每步的 token 数成正比，四个 rank 一样，和路由、提示词无关。所以崩溃不是某个 rank 的专家太忙占多了显存，也不是开重叠的逻辑：16,384 时同样的组合（模型自己的路由、开重叠、50 个词）跑了 50 秒没有出错。
- 「分配失败后重试」是 PyTorch 第一次要不到一整块，把缓存还给设备以后再要到的（worker 日志里的 `memory allocation failed with OOM`）。32,768 时刚重启的引擎上就有，说明这个预算在 0.85 下本来就贴着上限。
- 崩掉的那个引擎已经跑了 108 分钟、23 次变更、两千多步。rank 3 上三次重试和最后那次失败都在提示词换成 50 个词以后的几秒里（14:54:10–14:54:13），之前 50 秒随机 token 的提示词没有重试。我的推测：提示词的长度变了，一步里各行的 token 数跟着变，缓存里留着的块大小对不上，2 GiB 的一整块又要不到。另外三个 rank 在之前两次重载 `kernels.deepinfer` 时也都有过重试。2 GiB 正好是 `[32768, 64, 512]` 的 bf16 张量的大小（attention 的 query 是这个形状）。
- 步开始前已分配的 155.9 GiB 里，0.85 × 178.35 = 151.6 GiB 是权重和状态池，其余是按预算开的缓冲区和录的图。PyTorch 之外（CUDA 上下文、kernel 库、各 rank 共享的缓冲区）还有约 8.4 GiB。

### 怎么用

- 每步 32,768 个 token 不要配 `--mem-fraction 0.85` 长时间跑。调到 0.80 左右能多出约 9 GiB，这次没有试；PyTorch 报错里建议的 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 也没有试。
- 每步 16,384 个 token 在 0.85 下余量有 11 GiB，吞吐低约 5%。
- 默认的 `sched.token_budget` 还是 2048，没有动。

## 验证

- `dev-server`：pyright 0 错误；`tests/units`、`tests/scenarios`、`tests/ci` 438 个通过（没有碰框架本体，`tests/framework` 没跑）。新加一个单测：`test_tokens_in_another_type_than_i64_are_refused_before_their_bytes_are_read`。
- n1 的 GPU 1、2：`tests/gpu/test_ds1017_deepinfer.py` 93 个通过，507 秒。
- 引擎上的贪心对照，7 个提示词（3 个短的，4 个 1,201–3,601 个 token 的），每个单独发、7 个一起发、错开 0.12 秒发，各生成 64 个 token：
  - `lay_out()` 和 executor 用已提交的代码（另一份工作区，热改上去）对比现在的代码（再热改回来）：21 段文本全部相同。
  - 现在的代码冷启动以后对比热改上去的：单独发和错开发的 14 段全部相同；一起发的 7 段里有 1 段不同（3,601 个 token 的那条）。
  - 一起发的文本本身不总是稳定：刚冷启动时连跑三次全部相同，压测并恢复配置以后什么都不改连跑两次，有两段不同。我的推测是 7 条请求怎么分到 4 个调度域、和谁排在同一步，会随引擎之前跑过什么而变，没有去证实。长提示词单独发和一起发的文本不同，用已提交的 `lay_out()` 时也是这样，不是这次改出来的。
- 压测以后恢复配置，引擎健康，身份 `c3a50e9ecaa5d604`。它和冷启动时的 `4f69014392d9231d` 不同，是因为 `deepinfer.force_random_gate = false` 现在显式写在配置里（和 `holder.sleep` 第一次醒来一样）；两次恢复以后是同一个身份。

## 热改和冷启动

| 引擎 | 变更 | 用时 |
|---|---|---|
| 13:07 起的那个 | `apply --source`：`lay_out()` 和 executor 换成已提交的代码，再换回现在的代码（各重载 `layers.ds1017`、`models.ds1017`、`core.plan`、`executors.pytorch`、`core.worker`；`--dry-run` 的计划和实际一致） | 各 5 s |
| | `apply --config`：预算 32,768 加强制随机；关掉强制随机（都重载 `kernels.deepinfer`、重录图） | 60 s、64 s |
| | `apply --config`：`sched.overlap`（live 键） | 2 s |
| | `apply --config`：`sched.overlap`，三个 worker 已经退出 | `failed`，62 s |
| 冷启动 | `stop`，`serve` | 20.5 min |
| 15:23 起的这个 | 探针工作区加配置、换配置、撤掉探针并恢复配置 | 71 s、74 s、67 s |
| | 16,384 的五次配置变更（两次只改 `sched.overlap`） | 64 s、3 s、61 s、2 s、61 s |

提交 13 次，`failed` 1 次，冷启动 1 次。没有发现更新机制的问题。

## 留下的事

- 每步 32,768 个 token 的显存余量（上面「怎么用」）。
- 真实路由比强制随机一步多的约 75 ms：Antigravity 的 profile 里 MoE 从 264 ms 到 373 ms，应是各 rank 的专家负载不均，这个分解我没有复测；EPLB 带冗余槽位以后的 prefill 没量。
- 开重叠以后和 DeepInfer 还差的 2%，没有细分。
- 稳态 prefill 的压测脚本没有进仓库。
- n1 上 Antigravity 留下的 `probe-ws` 和 `/tmp` 里的临时文件、我留下的 `/tmp/devin-ab`、`devin-ref-ws`、`devin-probe-ws` 和崩掉的运行目录，都还在。
