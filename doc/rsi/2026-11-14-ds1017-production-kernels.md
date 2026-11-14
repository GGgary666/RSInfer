# DS 1017 的生产 kernel：怎么在 SM100 上接的，在跑着的引擎上验了什么（2026-11-14）

- 记录人：Claude Fable 5.2（Claude Code）。
- 对应：[../specs/v1-ds1017.md](../specs/v1-ds1017.md) 第 9.2、9.3 节；PR #65。前一份记录：[2026-11-14-ds1017-on-a-live-engine.md](2026-11-14-ds1017-on-a-live-engine.md)。
- 机器：SM100（任务服务，节点 n1，DeepInfer 的环境）；本地 4090 D 跑不了这些库（都要 SM 10.0），只跑了不依赖它们的测试。

## 结论

1. **`kernels.deepinfer` 接好了**：DeepGEMM、FlashMLA、TileKernels、DeepSelect 成了 14 个 `ds.*` definition 的 solution，另外 3 个 definition 在这个后端里是 torch 的写法。每个都有对参照的记录，都能录进 CUDA graph。
2. **在跑着的引擎上热改都提交了**，没有发现更新机制的问题，没有写 framework-bugs 报告：

   | 做了什么（真权重前 8 层的引擎，经 HTTP） | 结果 |
   |---|---|
   | 冷启动，`--cuda-graph auto` | prefill、decode、graph 三张表都选了 `deepinfer`；graph 桶 1、2、4、8 |
   | `apply --config '{"plan.cuda_graph": "off"}'` | 2–3 秒提交；文本和有 graph 时相同 |
   | `apply --config '{"dsv4.backend": "tilelang"}'` | 3 秒提交；文本变了（两套 kernel 算术不同） |
   | 两个键改回 `auto` | 3–4 秒提交；文本回到冷启动时的，吞吐也回到原来的数 |
   | 改 `kernels/deepinfer/ops.py` 的一行（结果不变），`apply --source` | 3 秒提交；文本和吞吐不变 |

   - 最后一行验的是带原生库的 kernel 单元重载：DeepGEMM 和 TileKernels 有进程级的设置（专家行的对齐），单元每次加载都设成同一个值，重载不需要重启进程。
3. **框架本体没有改。**
4. **这次开发的大部分不是在引擎上做的，是在 SM100 上跟分支的任务里做的。** 原因和做法见下一节。

## SM100 上怎么做到"改一下几十秒看到结果"

SM100 只能通过任务服务用，一个任务从提交到开始跑要几十秒到几分钟，引擎冷启动加真权重还要一两分钟。这些库本地又跑不了，所以每一次试探都得上 SM100。

- 写了 `tests/gpu/sm100_follow.sh`：任务起来以后跟着一个分支，每推一个提交，就把它解到一个目录里，在 DeepInfer 的环境里跑提交里的 `tests/gpu/sm100_follow_check.sh`，输出进任务日志。
- 一次往返（推提交到看到结果）是检查本身的时间加十几秒：kernel 的对照测试 40 秒，真实形状的引擎检查 2 分钟，真权重 8 层 4–8 分钟。
- 用一个不开 PR 的草稿分支（`claude/ds1017-kernels`）来推，不触发 CI；做完再合到 PR 的分支上。
- 前三个提交是探针：先看库函数的参数、布局和对齐要求，再写单元。探针回答了几件事：我们用 torch 排的 scale 布局和 DeepGEMM 自己转出来的逐位相同；DeepGEMM 的 FP8 乘法和我们按固定顺序求和的参照逐位相同；MoE 的差别来自官方实现多的一次 bf16 舍入。
- kernel 单元写完以后才换成 `sm100_session.sh`（起引擎、热改），做上面那张表。

和 AGENTS.md「先在引擎上做」的出入：kernel 对参照的比较不需要引擎，直接调函数最快；需要引擎的部分（选 kernel、录 graph、调度、HTTP）是在引擎上验的。在真前向上逐个比 kernel 时用的是 `check_ds1017_real.py --shadow`，它在 worker 栈里把层单元取 kernel 的那个方法包了一层，每次调用同时用官方 kernel 算一遍。这和"把检查临时写进单元"是一个意思，只是写在检查脚本里，没有改单元。

## 查到的几件事

- **FlashMLA 的 sparse kernel 要求每个 query 的 key 数是 128 的整数倍**，不是的话直接断言失败。封装里用 -1 补齐。探针里碰巧用了 640，到引擎上才撞到。
- **录 graph 时上下文是空的**：DeepSelect 的封装要先处理 0 个候选。
- **TileLang 的 kernel 第一次用到某个形状时要编译**（一个 2 秒左右）：真权重的引擎第一次 8 路并发时量出来只有 33 tokens/s，第二次是 1141 tokens/s。量吞吐前要先用同样的并发数跑一遍。
- **截到 8 层的真权重模型和随机权重的模型一样对末位敏感**（细节在规格 9.2）。所以整模型的对照只能等 40 层都放得下的时候。
- **本地全量测试有两次超时**，当时机器的 load 是 176（别的进程），单独重跑都过。

## 追加（2026-11-14 晚）：状态换成 DeepInfer 的字节布局

对应规格 9.3。做法和上面一样：本地改，CPU 测试和本地 GPU 上的逐位对照先过，再推到 SM100 的跟分支任务上，最后起引擎会话。

| 在跑着的引擎上（真权重前 8 层，经 HTTP） | 结果 |
|---|---|
| 冷启动，状态是字节布局，`--cuda-graph auto` | 起来了；decode 的 graph 里是 FlashMLA 的 decode kernel |
| 关 graph、切到 `tilelang`、两个键改回 `auto` | 都热提交（2–4 秒）；graph 和 eager 文本相同；切回去文本相同 |
| 后端 `tilelang` 的文本对比换布局之前那次会话 | 摘要相同（`3035fd4bac0a`）：换布局没有改变官方 kernel 这条路的任何一位 |

- 第一次起会话时引擎没起来：`core.plan` 录 graph 时 FlashMLA 报块的字节跨度超过 int32（规格 9.3 的"一个坑"）。会话脚本把各进程的日志留在了结果包里，从 `engine.log` 里直接看到了报错。
- 没有发现更新机制的问题。框架本体没有改。

## 追加（2026-11-14 夜）：DeepInfer 自己的算子和 DeepGEMM 的 indexer

对应规格 9.4。做法照旧：本地写参照和 TileLang 的组合、CPU 测试和本地 GPU 上对官方逐位，再到 SM100 的跟分支任务上对 DeepInfer 的算子，最后起引擎会话。

在跑着的引擎上（真权重前 8 层，经 HTTP，任务 `20261114T180148Z-af52ce33`）：

| 做了什么 | 结果 |
|---|---|
| 冷启动，`--cuda-graph auto` | 所有 definition 都选 `deepinfer`，graph 桶 1、2、4、8 |
| 发一个短提示词（"Hello"） | worker 失败：DeepInfer 的 `load_keys` 在 0 行时 reshape 出错。引擎停止服务，之后的请求回 400，说明里写着 worker 的错误 |
| 改 `kernels/deepinfer/ops.py`（0 行直接返回），`apply --source` | 3 秒提交，引擎恢复服务（身份变了，调度器清掉故障），没有重启 |
| 关 graph、切 `tilelang`、改回来 | 都热提交；graph 和 eager 文本相同；`tilelang` 的文本和这一轮改动之前相同 |

- 这是这个阶段第一次在 SM100 上靠热改修好一个让 worker 失败的 bug：会话任务的检查脚本可以随提交换，所以先推了一个只发一个请求、打印 worker 日志的检查，看到错误，再推修复。
- 撞到的几件事都是"DeepInfer 只走一种形状，我们走了另一种"：
  - top-k 不是 512 时，TileLang 的候选恢复要求行跨度是 8 的倍数，`convert_kv_index` 在 300 宽时编不出来；
  - TileLang 的 kernel 按宽度编译，eager decode 每步宽度不同就每步重编（9 秒一步）；
  - DeepGEMM 的稀疏 indexer 元数据在每个 stream 上第一次用时要清零一块工作区，graph 录制时不许做，所以要先在录制用的 stream 上热身（平台的 `warmup` 本来就是这样）。
- 没有发现更新机制的问题，框架本体没有改。
