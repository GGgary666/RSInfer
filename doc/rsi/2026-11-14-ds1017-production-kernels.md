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
