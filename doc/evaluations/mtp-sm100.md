# MTP：SM100 实测、两处修复与 SGLang profile 对照（2026-11-14）

记录人：Codex。审查基线为 PR #61 的 `ec8323121dca79065a187f15f48f0450dfd0ef1a`。

## 动机与设置

验证完整模型的真实 MTP 草稿能否带来加速，并检查 CUDA Graph、overlap 和前缀缓存的实际运行路径。用户随后授权修复发现的问题，并要求把修复版与 SGLang 的差距 profile 清楚。

- n1 / GPU 6，SM100（SM100、148 SM、约 180 GiB）；所有长任务在独立 tmux 中运行。
- 完整 64 层 Qwen3.8-27B 真权重，包含 checkpoint 原有的 15 个 `mtp.*` 张量；TP=1、BF16、FP32 recurrent state。
- PyTorch 2.11.0+cu130；SGLang 0.5.23，使用同一模型的已有 language-model-only checkpoint，MTP 权重也在其中。
- 双方采用 EAGLE 链式 `(num_steps, topk, num_draft_tokens)=(3,1,4)`，context=4096，page=16，token budget=2048，max rows=16，memory fraction=0.80，GDN/KV ratio=0.9，FlashInfer attention、Triton GDN、overlap 和 prefix cache 开启。prefill graph 关闭；SGLang 保留其原有 draft/verify/draft-extend 图路径。
- 正式吞吐：同一组中英文解释、代码、写作提示词，贪心，输出 256 token，并发 1/4/8，每客户端 4 请求；每档先预热。吞吐数据来自无 profile 探针的运行。
- Profile：另外采集 batch=1/8 的 CUDA event 分阶段计时，以及双方各 30 步的 PyTorch CPU/CUDA 时间线。Profile 负载输出 512 token，用来定位耗时；其受采集和导出影响的吞吐不进入正式对比表。

## 代码归属与本轮修改

**耗时约 9.8 ms 的状态提交路径来自 Claude 在 #61 原始实现中的 `engine/kernels/reference/spec.py::commit_states`。本轮没有修改这个函数。**

Codex 只改了两个生产文件：

1. `engine/core/plan/unit.py`：录 verify graph、记录依赖代数之前，初始化投机所需的 batch 缓冲区。
2. `engine/executors/pytorch/spec.py`：把接受长度的 D2H 拷贝与 CUDA event 放在 draft extend 之前，和 SGLang EAGLE v2 的发布顺序一致。

下文“修复版”指包含这两处改动的整个版本；其中的 PyTorch 状态提交实现仍然是 PR 原来的代码。没有修改框架本体，没有新增测试文件，也没有加入后续的 kernel 优化。

## 已修复的问题

### 刚录好的 verify graph 立即过期

原顺序是 `core.plan` 先录图并记录 batch generation=1，随后 executor 构造 `Speculative`，`batch.prepare_spec()` 增加缓冲区并将 generation 改成 2。

实际 SM100 冷启动日志显示已经录了 1/2/4/8/12/16 的 verify 图，但运行时：

```text
saved=[1, 0, 6, 6, 0, 0]
current=[2, 0, 6, 6, 0, 0]
valid=False
355 行 verify，replays=0
```

`auto` 全部退回 eager；`required` 则启动成功、第一条请求在第 2 步失败，HTTP 400：

```text
StaleGraph: no ready verify graph holds 1 rows, and graphs are required
```

原版关闭再打开投机后，batch 缓冲区已存在，重新录图会碰巧恢复有效。这也解释了冷启动和热切换的巨大性能差异。现有 GPU 用例只比输出和接受长度，eager fallback 也能通过该比较。

修复后，从全新进程直接以 `EAGLE + cuda_graph=required` 启动并完成全部请求；探针记录 `saved=current=[2,0,6,6,0,0]`，357 行单行 verify 对应 **357 次实际 replay**。

### 接受长度的通知晚于 draft extend

原代码在 draft extend 和 `_keep()` 之后才拷贝接受长度、记录事件，下一步的 `settle()` 会把这段 GPU 工作也等完。SGLang 的 `on_publish(new_seq_lens)` 位于 draft extend 之前。

修复将拷贝和事件提前到接受/状态提交之后、draft extend 之前。下一步 CPU 可以准备位置和 attention metadata；GPU 上的 draft extend 和下一步操作仍按同一 stream 的顺序执行。

修复版与原版“通过热切换使图有效”的同组 56 个请求，文本 **56/56 相同**。temperature=0.8、固定 seed 的重复请求一致；max_tokens=1/2/3/7/17 均正确截断；生成结果接入下一次提示词时，实际从 **256** 开始并带一份状态恢复。

## 修复版与 SGLang 的实际吞吐

单位：输出 token/s。

| 并发 | RSInfer 普通 decode | 原 PR 冷启动 MTP（退回 eager） | 原 PR 热切换后 MTP（图有效） | Codex 两处修复后 MTP | SGLang MTP |
|---:|---:|---:|---:|---:|---:|
| 1 | 93.2 | 65.3 | 151.3 | **157.3** | **177.2** |
| 4 | 361.0 | 238.0 | 479.0 | **492.4** | **543.2** |
| 8 | 685.7 | 422.7 | 732.9 | **746.0** | **1055.2** |

SGLang 不开投机时为 93.5/360.5/692.5 token/s，与 RSInfer 的普通 decode 基线接近。

修复版相对自己的普通 decode 提升 **1.69× / 1.36× / 1.09×**，达到 SGLang MTP 的 **88.8% / 90.6% / 70.7%**。并发 8 的剩余差距最明显。额外请求探针中，每行 verify 平均接受约 **2.86 token（含 bonus）**，真实草稿正常工作。

## Profile：优先优化原 PR 的状态提交

修复版的 CUDA event 分阶段中位数，单位 ms/verify step：

| 实际 batch | draft | target verify | accept | GDN commit | draft extend | 整步 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.576 | 12.074 | 0.040 | **1.604** | 0.684 | 17.032 |
| 8 | 2.676 | 14.231 | 0.046 | **9.793** | 0.690 | 27.477 |

batch=8 时，commit 占整步约 **36%**。相邻步之间的空档中位数约 **0.003 ms**。这里 CPU `submit` 的耗时包含等待接受长度的 CUDA event，不能直接当作 Python 计算耗时。

CPU/CUDA 时间线进一步确认：

- RSInfer 的 batch=8、30 个 verify 窗口中，通用 `index_elementwise` 和 `where` kernel 合计约 **294.9 ms**，折合 **9.83 ms/步**。分段标记 `mtp.commit` 的 GPU 区间约 10.0 ms/步，与 event 计时一致。
- SGLang 同样 30 步中的状态 scatter kernel 合计约 **16.59 ms**，折合 **0.55 ms/步**。使用的是 `_fused_mamba_state_scatter_with_mask_kernel` 和 `_fused_conv_window_scatter_multi_kernel`。
- batch=1 时，SGLang 相关 scatter 合计约 **0.070 ms/步**；RSInfer 的 commit event 计时约 **1.60 ms/步**。

原因直接对应原 PR 的 `commit_states`：它对每个池先做一次 `inter[:, lane, step]` gather，再写回；随后无论是否跨跟踪边界，都求值 `inter[:, lane, track.clamp_min(0)]` 和 `pool[:, dst]`，经过 `torch.where` 后再写回。没有跨边界的普通步也会搬这些大张量。64 层模型的一行 GDN 状态约 147 MiB，8 行的一份状态就是约 1.15 GiB，多次临时 gather 和写回的代价很大。

**下一步优先搬 SGLang 的融合状态 scatter/commit kernel。** 直接从选中的中间 step 拷到运行槽；`track_steps < 0` 的行跳过跟踪拷贝，避免构造整份临时状态和重复读取 padding 槽。对应上游文件为 `python/sglang/kernels/ops/mamba/mamba_state_scatter_triton.py`。这是 Claude 原 PR 当前使用参照实现留下的性能瓶颈，与 Codex 的图初始化、事件发布两处修复分开。

其后再优化草稿侧 CPU 发射：profile 中 batch=8 的 draft CPU 区间中位数，RSInfer 约 3.08 ms、SGLang 约 1.27 ms；draft extend 分别约 1.93/1.22 ms。PR 已说明这两段暂时 eager，后续可按 SGLang 补 graph。当前应先处理状态提交这个更大的、随 batch 增长的损耗。

## 验证与记录位置

- 原 PR 的 CPU 全量：678 passed、2 项达到单测 120 秒超时；两项单独串行复跑均通过（33.21 秒）。全量 Pyright 0 错误。修复后两处文件的 Pyright 0 错误。
- 用户随后要求以现场验证修复为主，因此没有新增测试或再次跑完整套件。上面的冷启动 required、请求检查、正式吞吐和 profile 都在 SM100 真模型上完成。
- n1：`/workspace/pr61_codex_ec83231/`。`repo/` 是原 PR，`fixed/` 是两处修复，`profile/` 是额外计时探针；后者不进仓库。
- `out/` 保存正式 benchmark JSON、请求输出、日志和 profiler trace；`tools/` 保存启动、请求和采集脚本。
- 时间线：`out/rsinfer-trace-bs{1,8}.json.gz`；`out/sg-traces/sg-bs{1,8}-mtp-bs{1,8}-TP-0.trace.json.gz`，可在 Perfetto 打开。
- 数据和脚本包：n1 的 `results.tar.gz`；`s3://<artifact-bucket>/rsinfer/rsinfer-pr61-codex/results.tar.gz`；本地 `/tmp/rsinfer-pr61-sm100-results/`。
- 所有本轮服务和 tmux 已结束，GPU 6 已释放。没有改动其他任务。
