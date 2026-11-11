# 真 kernel 在 SM100 上的结果：完整 64 层模型对原生 SGLang（2026-11-11，Gemini 测）

**这份记录里的数字都不是我测的。** 它们来自 Gemini 4 Carbon（Antigravity）在 #51 上的两条评论，我（Claude Fable 5.2）按用户的要求原样搬过来，方便和 4090 D 上的两份记录放在一起看：

- 第一轮，`2be83ed`（第一批，[real-kernels-4090d.md](real-kernels-4090d.md)）：[评论](https://github.com/open-agent-infra/agxinfer/pull/51#issuecomment-6793605157)
- 第二轮，`a471a04`（第二批，[real-kernels-fused.md](real-kernels-fused.md)）：[评论](https://github.com/open-agent-infra/agxinfer/pull/51#issuecomment-6795957652)

最后一节"我的备注"是我读这些结果时的两点保留，和 Gemini 的结论分开写。

## 结论（Gemini 的）

1. **64 层上，HTTP decode 吞吐从 SGLang 的 63.5–72.1% 升到 92.7–100.9%。** 1 并发热态 94.3%，64 并发 100.9%。每步比 SGLang 多出的时间从 4.07 ms 降到约 0.54 ms。
2. **HTTP prefill 吞吐从 74% 升到 97.7–100.2%。** 1 并发的首 token 时间 102.7 ms，SGLang 是 103.1 ms。
3. **decode 剩下的约 5% 全在设备一侧**，host 上每步的准备被重叠完全盖住（步与步之间 GPU 空闲 3 µs）。
4. **2000 token 的 prefill 不走 graph 也追平了**：host 发完 64 层要 59.7 ms，比设备算完的 97.9 ms 短。
5. 第二批的代码复审没有发现新的功能、数值或框架边界问题；SM100 上的 PDL 路径都通过。

## 环境和配置

- **机器和模型**：单卡 NVIDIA SM100（`sm_100`，183 GiB），TP=1。完整的 64 层 Qwen3.8-27B checkpoint：48 层 GDN 加 16 层 full attention，bf16，SSM 状态 fp32，`context_len = 4096`。
- **负载**：`tests/gpu/bench_serving.py`，流式、贪心、`ignore_eos`，prompt 全程唯一，每档并发先发一轮不计时的预热请求。
  - decode：`--prompt-words 32 --max-tokens 256 --requests 4 --concurrency 1 4 8 16 32 64`
  - prefill：`--prompt-words 2000 --max-tokens 8 --requests 8 --concurrency 1 4 8`
- **三组对比**（全部并发都是 CUDA graph 对 CUDA graph）：

| | 命令或配置 | 说明 |
|---|---|---|
| agxinfer，第一轮 | `deploy/qwen38.py serve --cuda-graph required --overlap --max-rows 64 --max-context 4096 --graph-buckets 1 2 4 8 16 24 32 48 64 --token-budget 8192 --mem-fraction 0.80` | 状态池 92.44 GiB：GDN 槽 305，KV 页 49,879（79.8 万 token）；`kept_bytes` 672 MiB（含 FlashInfer 的 384 MiB workspace） |
| agxinfer，第二轮 | 同上，但不传 `--graph-buckets` | 默认就录齐 1、2、4、8、12、16、24、32、40、48、56、64；`kept_bytes` 266.0 MiB（278,921,216 B），冷启动和热重载一致 |
| SGLang，`fi+triton` | `--attention-backend flashinfer --linear-attn-backend triton --max-running-requests 64 --context-length 4096` | 和本 PR 同一族 kernel。decode graph 录同样的 12 档；另外默认录 74 档 prefill graph（约 46 秒，约 4.4 GiB 显存）；`chunked_prefill_size=16384` |
| SGLang，默认 | 不指定 backend | 在 `sm_100` 加 head dim 256 时 attention 自动选 `triton`；`mem_fraction_static=0.859`，GDN 槽 337，KV 88.3 万 token |

第二轮里 SGLang 的两组也用新版压测脚本重跑了，和第一轮相差不到 0.5%。

## 测试

| | 第一轮（`f096075`） | 第二轮（`a471a04`） |
|---|---|---|
| `uv run pyright`、`check_tree` | 0 错误 | 0 错误 |
| CPU 默认测试（`pytest -n 8`） | 596 通过 | 597 通过 |
| `tests/gpu/test_real_kernels.py` | 9 通过，2 失败 | 22 通过（含 SM90+ / SM100 的 PDL 路径和 5 个 review 修复的回归用例） |
| 其余 GPU 测试文件 | 全部通过 | 全部通过：`test_qwen38_partial` 4，`test_cuda_graph` 9，`test_cuda_graph_adversarial` 3，`test_cuda_graph_apply` 4，`test_state_rebuild` 1，`test_qwen38_overlap` 1 |

- 第一轮失败的两个用例是容差和测试写法的问题，第二批修了（见 real-kernels-fused.md 的"review 的修复"）。
- 第二轮把新增的 4 个 Triton kernel 文件和上游 SGLang `e838360` 逐行比过，kernel 本体和 launcher 一致。
- bf16 下 4 层模型和 HF 的逐层余弦是 0.9997（第一轮）。

## HTTP decode（64 层，prompt 32，生成 256）

吞吐是输出 token/s，每 token 间隔是中位数。

| 并发 | SGLang 默认 | SGLang `fi+triton` | agxinfer 第一轮 | agxinfer 第二轮 | 第二轮相对 `fi+triton` | 相比第一轮 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.0 / 10.04 ms | 97.0 / 10.14 ms | 64.5（热态 68.6）/ 15.22（14.22）ms | 89.9（热态 91.5）/ 10.87（10.68）ms | 92.7%（热态 94.3%） | +39.4%（热态 +33.4%） |
| 4 | 373.3 / 10.37 ms | 370.7 / 10.48 ms | 252.5 / 15.50 ms | 351.4 / 11.13 ms | 94.8% | +39.2% |
| 8 | 714.4 / 10.85 ms | 709.7 / 10.93 ms | 469.1 / 16.36 ms | 669.3 / 11.44 ms | 94.3% | +42.7% |
| 16 | 1,343.7 / 11.58 ms | 1,334.0 / 11.63 ms | 850.9 / 18.11 ms | 1,261.9 / 12.18 ms | 94.6% | +48.3% |
| 32 | 2,377.1 / 12.79 ms | 2,393.7 / 12.79 ms | 1,565.4 / 19.77 ms | 2,262.6 / 13.58 ms | 94.5%（对默认 95.2%） | +44.5% |
| 64 | 3,687.7 / 15.67 ms | 3,697.3 / 15.60 ms | 2,655.3 / 22.74 ms | 3,730.1 / 16.30 ms | 100.9%（对默认 101.1%） | +40.5% |

首 token 时间的中位数（这一组的 prompt 只有 32 个 token）：

| 并发 | 1 | 4 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|
| SGLang `fi+triton` | 51.5 ms | 97.0 ms | 99.8 ms | 106.1 ms | 157.5 ms | 409.4 ms |
| agxinfer 第二轮 | 73.9 ms | 72.2 ms | 131.8 ms | 130.7 ms | 171.6 ms | 210.8 ms |
| agxinfer 第一轮 | 107.2 ms | 102.4 ms | 190.2 ms | 188.2 ms | 203.0 ms | 289.3 ms |

- "热态"：1 并发那一档在刚启动时测到的比稳定以后低一些，两个数都列了。
- **第一轮里默认桶只到 8 行的后果**：16 / 32 / 64 并发退回 eager 以后，吞吐从 850.9 / 1,565.4 / 2,655.3 跌到 258.0 / 509.9 / 996.1 token/s，慢 2.7–3.3 倍。第二批因此改了默认的桶。

## HTTP prefill（64 层，prompt 2000，生成 8）

吞吐是输入 token/s，后面是首 token 时间的中位数。`tb` 是 `--token-budget`。

| 并发 | SGLang 默认 | SGLang `fi+triton` | agxinfer 第一轮（`tb=8192`） | agxinfer 第二轮（`tb=8192`） | agxinfer 第二轮（`tb=16384`） | 第二轮（`tb=16384`）相对 `fi+triton` |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,914 / 106.1 ms | 11,220 / 103.1 ms | 8,308 / 138.2 ms | 10,894 / 103.8 ms | 10,962 / 102.7 ms | 97.7%（对默认 100.4%） |
| 4 | 16,988 / 386.3 ms | 17,289 / 381.0 ms | 12,866 / 504.4 ms | 16,936 / 291.8 ms | 16,934 / 293.4 ms | 97.9%（对默认 99.7%） |
| 8 | 18,997 / 748.3 ms | 19,350 / 738.6 ms | 14,470 / 885.9 ms | 19,257 / 534.0 ms | 19,386 / 728.1 ms | 100.2%（对默认 102.0%） |

- `tb=16384` 是为了和 SGLang 的 `chunked_prefill_size` 对齐。
- 第一轮里 `tb=2048` 时是 8,354 / 138.0 ms、12,525 / 429.7 ms、13,747 / 691.1 ms。

## 进程内的步级拆解（第二轮，64 层）

### prefill 一步（eager）

| 配置 | host 提交 | 设备执行 | 端到端 | 这一步的吞吐 | 16 层 KV 写入的设备时间 |
|---|---:|---:|---:|---:|---:|
| 1 行 × 2000 token | 59.72 ms | 97.92 ms | 98.03 ms | 20,401 token/s | 0.34 ms（0.35%） |
| 4 行 × 2000 token | 83.25 ms | 368.13 ms | 368.35 ms | 21,719 token/s | 1.09 ms（0.30%） |

Gemini 的解释：

- Python 把 64 层全部入队只要 59.7 ms，比设备算完 64 层的 97.9 ms 短。从第 0 层的第一个 GEMM 发出去起 GPU 就在满载，host 发完时 GPU 大约算到第 39 层。露在关键路径之外的 host 开销只有 98.03 − 97.92 = 0.11 ms。
- 第一轮慢到 138.2 ms，是因为 64 层有 4900 多次没融合的 launch，把 host 提交推到 100 ms 以上，同时设备上多出约 35 ms 的小算子时间。
- 4090 D 上 1 并发 prefill 仍慢 8 ms，是因为那里只有 4 层，一步的设备时间短，Python 发 kernel 的尾巴露在外面。
- 所以在 64 层的 SM100 上不需要 prefill graph，还省了 SGLang 录 74 档 prefill graph 的 46 秒和 4.4 GiB 显存。

### decode 一步（graph 回放，`depth=2`，`ctx=512`）

| batch | 每步中位数 | 进程内吞吐 | host 提交 | 设备执行 | 步间 GPU 空闲 | 16 层 KV 写入的设备时间 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.73 ms | 93.2 token/s | 0.84 ms | 10.69 ms | 0.003 ms | 0.24 ms |
| 4 | 11.15 ms | 358.8 | 0.88 ms | 11.10 ms | 0.003 ms | 0.25 ms |
| 8 | 11.44 ms | 699.1 | 0.92 ms | 11.41 ms | 0.003 ms | 0.23 ms |
| 16 | 12.29 ms | 1,302.1 | 0.97 ms | 12.17 ms | 0.003 ms | 0.25 ms |
| 32 | 13.39 ms | 2,390.5 | 1.10 ms | 13.30 ms | 0.003 ms | 0.25 ms |
| 64 | 15.50 ms | 4,128.0 | 1.33 ms | 15.31 ms | 0.003 ms | 0.25 ms |

Gemini 的解释：

- 融合路径把 1 并发热态的每 token 间隔从 14.22 ms 压到 10.68 ms，一步省了 3.54 ms，是第一轮 4.07 ms 差距的 87%。
- host 上每步的 `prepare` 和 plan 合计 0.84–1.33 ms，开重叠时被上一步 10.7–15.3 ms 的设备执行完全盖住。
- 1–32 并发剩下的约 0.54 ms（进程内设备时间 10.69 ms，SGLang 是 10.14 ms）全在设备一侧，Gemini 把它分成三项：

| 项 | 估的大小 | 说明 |
|---|---:|---|
| 16 层的 KV 写入（两次 `index_put`） | 0.18–0.20 ms | SM100 上 16 层合计 0.23–0.25 ms；SGLang 的单个 `store_kvcache` 约 0.05 ms |
| graph 外每步一次的准备 | 0.08–0.10 ms | 拼 KV 槽列表等操作比 SGLang 的一个 Triton kernel 多 6 次 launch |
| 48 层 GDN 输入投影的双流并发 | 约 0.24 ms | SGLang 在录 graph、且 token 数小于 1024 时，把很小的 `in_proj_ba` GEMM 放到另一个 stream 上和 `in_proj_qkvz` 并行（`qwen3_5.py` 的 `_gdn_use_alt_stream`）；这里在单个 stream 上串行，每层约多 5 µs |
| 合计 | 约 0.52 ms | |

- 64 并发时跨进程的 HTTP、调度和 SSE 比进程内多约 0.8 ms，但端到端吞吐（3,730.1 token/s）已经超过 SGLang（3,697.3）。

## 4 层 microbench（SM100，`tests/gpu/bench_cuda_graph.py`，`max_context` 4096）

每步毫秒，括号里是 graph 的 token/s。

| batch | depth | 第一轮 eager / graph | 第二轮 eager / graph | 第二轮 host 提交 eager / graph | 第二轮设备执行 eager / graph |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 4.87 / 2.00（501.0） | 3.28 / 1.75（570.1） | 2.91 / 0.80 ms | 3.19 / 1.62 ms |
| 1 | 2 | 4.69 / 1.40（714.4） | 3.09 / 1.18（850.1，+19.0%） | 2.91 / 0.79 ms | 3.09 / 1.16 ms |
| 4 | 1 | 5.14 / 2.16（1,850.6） | 3.37 / 1.86（2,147.6） | 2.97 / 0.80 ms | 3.29 / 1.74 ms |
| 4 | 2 | 4.99 / 1.51（2,656.7） | 3.19 / 1.21（3,317.7，+24.9%） | 2.99 / 0.82 ms | 3.19 / 1.19 ms |
| 8 | 1 | 5.32 / 2.29（3,490.2） | 3.47 / 1.97（4,062.1） | 3.07 / 0.82 ms | 3.39 / 1.85 ms |
| 8 | 2 | 5.10 / 1.56（5,129.5） | 3.30 / 1.24（6,470.6，+26.1%） | 3.11 / 0.85 ms | 3.30 / 1.22 ms |

- `max_context` 256 时第二轮 `depth=2` 的 graph 是 1.16 / 1.20 / 1.24 ms，和 4096 一样（第一轮是 1.40 / 1.50 / 1.56 ms，也一样）。
- 两轮的 12 组用例 graph 和 eager 的 token 都相同。
- 对照 4090 D 上同样的 8 行、`depth=2` graph：第一批 6.40 ms，第二批 6.19 ms。那里一步压在 GEMM 上，SM100 上没有。

## 其他（第一轮）

- graph 留下的显存：桶 1、2、4、8 时 557,842,432 B（532 MiB，含 384 MiB workspace；4 层和 64 层报的是同一个数）；64 层、9 个桶到 64 行时 704,643,072 B（672 MiB）。第二轮 64 层、12 个桶是 266.0 MiB。
- SGLang 在 SM100 上默认的 `triton` attention 和显式指定的 `flashinfer` 吞吐相差不到 2%，差距不在 attention 和 GDN 后端的选型上。

## 我的备注

这两点是我读结果时的保留，没有在 SM100 上验证过。

1. **"不需要 prefill graph"只对长 prompt 成立。** decode 那一组的 prompt 只有 32 个 token，首 token 时间在低并发下比 SGLang 高 20–30 ms（1 并发 73.9 对 51.5 ms，8 并发 131.8 对 99.8 ms）。按上面的拆解，eager prefill 一步的 host 提交约 60 ms，它跟 launch 的次数走，不随 prompt 变短而变短；32 个 token 的设备时间远小于这个数。所以短 prompt 的首 token 时间压在 host 上，prefill graph（或者每层更少的 launch）在这里还能省 20 ms 左右。
2. **剩下 0.54 ms 的三项里，只有 KV 写入一项看起来是量的**，而且评论里没说是怎么量的（回放的 graph 里放不了计时点）。另外两项是估的，三项加起来等于 0.54 不能说明每一项都对。双流那一项的前提我对着 SGLang 的源码核过，是真的；每层 5 µs 这个数没有依据。要确认，可以在 SM100 上给两边各抓一份回放步的 profiler trace 按 kernel 比，或者做掉其中一项再量。
