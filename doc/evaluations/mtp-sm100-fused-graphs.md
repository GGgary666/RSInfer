# MTP：融合状态提交与 draft graph 的 SM100 评估（2026-11-14）

记录人：Codex。评估版本为 PR #61 的 `0e4f7b5aa7a46b71a693e78c573c2722adfdcca5`。第一轮修复和测量见 [mtp-sm100.md](mtp-sm100.md)，本轮对应 [GitHub 复审评论](https://github.com/open-agent-infra/rsinfer/pull/61#issuecomment-6872981962)。

**结论：本轮没有发现新增的正确性阻塞项。两处原有修复保留正确；这次 fused GDN commit 和 draft / draft-extend graph 的性能收益在完整模型上成立。**

## 动机与设置

本轮验证 Claude 新增的融合状态提交、两类 draft graph 和投机缓冲区重建，并补齐同一批提示词下 SGLang 的接受长度及分阶段 profile。

- n1 / GPU 6，SM100，完整 64 层 Qwen3.8-27B 真权重及其 MTP 权重，TP1、BF16、FP32 recurrent state。
- 双方采用 EAGLE `(steps, topk, draft_tokens)=(3,1,4)`；context=4096、page=16、max rows=16、chunk/token budget=2048、memory fraction=0.8、GDN/KV ratio=0.9，FlashInfer attention、Triton GDN，overlap 和 prefix cache 开启。
- 相同中英文提示词，输入 29–38 token、输出 256 token，temperature=0、ignore_eos=true，并发 1/4/8，每客户端 4 请求。RSInfer 为 `cuda_graph=required`。
- RSInfer：PyTorch 2.11.0+cu130 / Triton 3.6.0；原生 SGLang 0.5.23：PyTorch 2.14.1 / Triton 3.8.0；双方 FlashInfer 0.7.2。SGLang 使用原权重的符号链接和 `language_model_only=true` 配置，未修改其实现。
- 正式吞吐不启用 GPU profiler；RSInfer 仅加轻量请求接受长度和 graph 回放计数。Profile 另跑输出 512 token 的请求，每档采集 30 步 CPU/CUDA 时间线。

SGLang 第一轮有一条 786 ms TTFT 的首次运行慢请求。最终比较使用与 RSInfer 相同的启动后 smoke 预热，再运行同一 benchmark 的结果。

## 吞吐与接受长度

吞吐单位为输出 token/s。“上一版”是包含 Codex 两处修复、尚未加入本次融合 commit 和 draft graph 的 `1bd2593`。

| 并发 | RSInfer 上一版 MTP | RSInfer 本版 MTP | SGLang MTP | RSInfer 普通 decode | SGLang 普通 decode |
|---:|---:|---:|---:|---:|---:|
| 1 | 157.3 | **190.9** | 178.3 | 92.2 | 93.9 |
| 4 | 492.4 | **655.4** | 591.4 | 359.6 | 362.2 |
| 8 | 746.0 | **1230.2** | 1156.5 | 674.4 | 693.4 |

本版相对上一版提升约 **21% / 33% / 65%**；相对本轮 SGLang 高 **7% / 11% / 6%**。这是上面这组短输入、输出 256 token 的端到端结果。

| 并发 | RSInfer 平均接受长度 | SGLang 平均接受长度 | RSInfer TTFT / TPOT（ms） | SGLang TTFT / TPOT（ms） |
|---:|---:|---:|---:|---:|
| 1 | 2.736 | 2.729 | 78.8 / 4.93 | 118.3 / 5.16 |
| 4 | 2.800 | 2.757 | 90.8 / 5.48 | 133.1 / 6.03 |
| 8 | 2.790 | 2.764 | 92.4 / 6.15 | 126.3 / 6.44 |

接受长度统一为 **每行 verify 接受的 token 数，含 bonus**，用总接受数除以总行 verify 数。RSInfer 在请求表采样前计数；SGLang 读取 `return_spec_tokens_details` 返回的 `spec_num_correct_drafts` 和 `spec_verify_ct`，计算 `(correct_drafts + verify_ct) / verify_ct`，均排除预热请求。两边接受长度很接近。

## Profile：原来的 commit 瓶颈已消除

RSInfer CUDA event 中位数，取 trace 导出后恢复正常运行的步骤，单位 ms/step：

| 实际 batch | draft | target verify | accept | GDN commit | draft extend | 整步 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1.474 | 11.616 | 0.038 | **0.166** | 0.649 | 13.996 |
| 8 | 1.591 | 13.811 | 0.046 | **0.629** | 0.651 | 16.768 |

batch=8 的 commit 从上轮 **9.793 ms 降至 0.629 ms**，整步从 **27.477 ms 降至 16.768 ms**。这次优化属于 Claude 新增的 fused scatter / draft graphs；上轮 9.8 ms 来自原来的 reference commit，Codex 两处修复没有改过它。

下面用双方各 30 步的 CUPTI 时间线比较 GPU 阶段中位数，单位 ms。SGLang verify 的多 stream 标记按同一步合成首尾时间区间；commit 取首个到最后一个 state scatter，accept 取 verify 结束到首个 state scatter。

| 实际 batch | 实现 | draft | target verify | accept | commit | draft extend |
|---:|---|---:|---:|---:|---:|---:|
| 1 | RSInfer | 1.859 | 12.775 | 0.037 | 0.177 | 0.644 |
| 1 | SGLang | 1.857 | 11.774 | 0.047 | 0.087 | 0.685 |
| 8 | RSInfer | 1.921 | 14.924 | 0.043 | 0.634 | 0.643 |
| 8 | SGLang | 1.938 | 13.947 | 0.056 | 0.567 | 0.683 |

进一步只加总 state scatter kernel 的设备时间，RSInfer batch=1/8 是 **0.067/0.525 ms**，SGLang 是 **0.070/0.551 ms**。融合搬运本身已经到达同一水平；RSInfer commit 阶段剩余约 0.1 ms 在 kernel 外的准备和发射间隙。

CPU 标记的中位数也确认 draft graph 生效：batch=8，RSInfer draft / draft extend 从上轮 **3.08/1.93 ms** 降为 **1.45/0.40 ms**；SGLang 本轮为 **1.26/1.21 ms**。当前 RSInfer target verify 占整步约 **82%**。后续继续提速，应优先看 target verify 的 forward 和 metadata 准备，commit 已经不是首要优化项。

## 代码与实际路径复核

- `prepare_spec` 在录图前完成，且 shape/dtype 相同才复用；参数变化时重新分配并更新 generation，逻辑正确。
- draft forward 每次 replay 前更新对应的 attention metadata；draft extend 使用刚完成的 verify 图的 hidden、positions 和 metadata，当前调用顺序中间没有另一次 attention prepare，配对成立。
- fused scatter 对照本地 SGLang 原实现检查过。这里的 intermediate conv 是 dense layout，使用同一 scatter 搬 conv 和 temporal 与该布局相符；未跨跟踪边界的行跳过额外拷贝。
- **未插探针前的全新冷启动**直接使用 `required`，单请求和并发请求均成功。随后加计数探针跑正式请求，记录到 **1532 次 verify、3064 次 draft、1532 次 draft extend replay**，D=4 下的比例正确，依赖 generation 有效。
- max_tokens=1/2/3/7/17 截断正确；temperature=0.8 固定 seed 重复输出一致；300 token 生成后接续提示词请求完成。
- 本地相关 CPU 用例 **25 passed**，本次变动相关模块 Pyright **0 errors**。完整模型验证在 SM100 完成，本轮没有重跑全量 CPU/framework/formal。

## 数据与记录

原始结果、脚本、请求文本和双方 trace：

- n1：`/workspace/pr61_round2_0e4f7b5/`，本轮服务和 tmux 均已结束。
- 包：`s3://<artifact-bucket>/rsinfer/rsinfer-pr61-round2/results.tar.gz`。
- 正式文件：`out/rsinfer_round2_mtp.json`、`out/sglang_round2_mtp_warmed.json`；时间线在 `out/rsinfer-trace-bs{1,8}.json.gz` 和 `out/sg-traces/`。
