# CUDA Graph：三份 review 之后的修改和测量（2026-11-10）

PR #45 收到三份 review：Gemini（代码）、Claude Opus 5.5 / Devin（SM100，完整 27B）、Claude Fable 5.2（RTX 4090 D，27B 真实结构的前 4 层）。这里记录按它们改了什么，以及在 4090 D 上改之前和改之后的测量。设计上的结论写在 [cuda-graph.md](../specs/cuda-graph.md) 的「实现选择」。

环境：RTX 4090 D 24 GB，PyTorch 2.11.0+cu130；模型是 `deploy/qwen38.py checkpoint --layers 4` 生成的 bf16 随机权重（3 个 GDN 层、1 个 attention 层，完整词表）。**完整 27B 和 SM100 上没有重新测**，那边改之前的数字在 Opus 的 review 里。

## 改了什么

| 问题（谁提的） | 改法 |
|---|---|
| 默认部署起不来：建图时把全部 `max_rows` 个 scratch 的状态克隆两遍再比较（三份都提了） | 每个桶只做一次 eager forward、capture、回放，比较 logits；不比较状态 |
| 每个 lane 一份 scratch，状态池少约 10%，状态单元去读 plan 的配置（Opus、Fable） | 照 SGLang，填充的 lane 共用 padding 行和 padding 项；`state.kv_paged`、`state.gdn_slots`、`core.state.holder` 恢复成 master 的样子 |
| 每个桶一张 `[bucket, max_context]` 的表，每步整表拷贝；scratch 下标每步重算（Gemini） | 固定 buffer 只剩一维的行下标、长度和写入位置，`req_to_token` 用持久 batch 的表 |
| 固定容量的 attention 在 `cuda_graph` 平台上连 eager 也用，batch 大或上下文长时比参照实现慢（Opus、Fable） | 放进单元 `kernels.torch_graph`，`requires=[CAPTURE]`，只有回放的步用；eager 用参照实现。不展开 GQA |
| 每个桶一个私有池（Opus）；预算量的是瞬时峰值，不是图留下的显存（Fable） | 一个 recorder 一个池，从最大的桶录起；预算量 `memory_reserved` 的增量 |
| 每次重建泄漏一组私有池，kernel 热改几次后 OOM，回滚也失败（Fable）；`mem_get_info` 预检会误报（Gemini） | 关闭图以后 `empty_cache()`；去掉预检 |
| `auto` 下预算不够或 kernel 不可捕获时启动失败（Opus、Gemini） | `auto` 走 eager 并在日志里写原因，`required` 才失败 |
| 依赖守卫每步对每个桶各算一次，再加两次（Opus、Fable） | 每步在 `buckets()` 里算一次；每个提供者一个 generation，不再每层一个 |
| `rebuild_graphs` 在 graph 平台上走不到，在 `torch_eager` 上让失败的 kernel 变更回滚不了；planner 写死平台名（三份都提了） | 删掉这个 action；派生产物没有 rebuild action 时 `core.deps` 重载它的 owner，不看平台 |
| executor 用 `getattr` 读契约外的 `required`；`Catalog.selected(clear=True)` 有副作用（Opus、Fable） | `required` 进 `PlanService`；`selected` 换成只在作用域内记录的 `recording()` |
| `Dependencies(debug=...)` 从不开启；建图时 `prepare` 跑两遍；executor 为回放的步也传 `last_index`；每步清零整个输入 buffer（Fable、Gemini） | 删掉或改掉 |
| 部署默认 `auto`（Opus、Fable） | 默认 `off`；默认的桶不超过 8 行 |

没有改的：固定容量 attention 的代价仍然随 `max_context` 增长（见下表 16384 一行），输出的位仍然随表宽变化。这是这种写法本身的性质，等 M2b 的 paged decode kernel。

## 测量

`tests/gpu/bench_cuda_graph.py`，8 行，桶 1/2/4/8，prompt 32，预热 16 步，测 128 步；ms / 步，overlap 关 / 开。eager 是 `torch_eager` 加参照 kernel（`--cuda-graph off` 的部署），不随 `max_context` 变。改之后是 2 次的中位数；改之前的 4096 用 9 GiB 预算才录得出四个桶，16384 录不出四个桶。

| `max_context` | batch | eager | graph，改之前 | graph，改之后 | 改之后相对 eager |
|---:|---:|---:|---:|---:|---:|
| 256 | 1 | 9.20 / 6.83 | 7.17 / 6.31 | 6.83 / 6.27 | 1.35× / 1.09× |
| 256 | 8 | 10.88 / 9.07 | 8.75 / 7.68 | 8.13 / 7.38 | 1.34× / 1.23× |
| 4096 | 1 | 9.23 / 6.84 | 7.89 / 7.01 | 6.85 / 6.30 | 1.35× / 1.09× |
| 4096 | 4 | 10.11 / 7.99 | | 7.89 / 7.27 | 1.28× / 1.10× |
| 4096 | 8 | 10.91 / 9.07 | 14.36 / 13.27 | 8.91 / 8.15 | 1.22× / 1.11× |
| 16384 | 1 | 9.17 / 6.84 | 10.28 / 9.45（单桶） | 7.02 / 6.48 | 1.31× / 1.06× |
| 16384 | 4 | 10.10 / 7.99 | | 9.24 / 8.61 | 1.09× / 0.93× |
| 16384 | 8 | 11.03 / 9.04 | 录不出 | 11.48 / 10.74 | 0.96× / 0.84× |

图留下的显存（四个桶合计，`core.plan` 日志的 `kept_bytes`）和 host 提交时间：

| | 改之前 | 改之后 |
|---|---:|---:|
| `max_context` 256 / 4096 / 16384 留下的显存 | 614 / 5376 MiB / 录不出 | 186 / 358 / 934 MiB |
| 建图，每个桶 | 0.05–0.59 s | 0.02–0.08 s |
| graph 一步的 host 提交（中位） | 1.03–1.20 ms | 0.70–0.82 ms |
| 64 行时状态池（GDN 槽 / KV 页，holder 按默认 0.85 分） | 573 / 93509 | 656 / 107020，与不开 graph 相同 |

- `deploy/qwen38.py serve --cuda-graph auto`，其余默认（64 行、4096）：改之前 `graph bucket 1 needs 2126726656 bytes`，起不来；改之后起得来，桶 1/2/4/8，留下 378 MiB。
- `serve --cuda-graph auto --max-rows 8 --max-context 1024`，默认 `mem_fraction`，连续给 `kernels/reference/ops.py` 加注释再 `apply`：改之前第 1 次就 OOM，回滚失败，`recovery_required`；改之后 10 次都提交，nvidia-smi 一直是 21152 MiB，每次 execute 约 0.13 s。
- worker 进程内重载 `kernels.reference` + `core.plan`：改之前 `memory_reserved` 每次涨 1.36 GiB；改之后 10 次不变（`tests/gpu/test_cuda_graph.py`）。
- CPU 引擎上去掉一个 decode 要用的 definition 再 `apply`：改之前 `change.abort_failed`，`recovery_required`；改之后 `aborted`，继续服务，下一个变更能提交（`tests/scenarios/test_m2a.py`）。

graph 和 eager 现在跑的是两个 attention kernel。fp32 下两者的 greedy token 都与 HF 相同（`tests/gpu/test_qwen38_partial.py`，12 个 token，3 个 prompt）；bf16 随机权重下跑 128 步，两者的 token 流在每个配置上都分开了，benchmark 的 `tokens_equal` 记录了这一点。真权重下没有测。

## 跑过的

- `uv run pyright`；`uv run pytest -n 8`。
- `tests/gpu/test_cuda_graph.py`、`test_cuda_graph_adversarial.py`、`test_cuda_graph_apply.py`、`test_qwen38_partial.py`。
- 没有跑：formal（这轮没有动 `unit.runtime.host`、`core.change`、`core.supervisor`、`core.nodeagent`）、`tests/gpu/test_qwen38_serve.py`（这台机器上没有 Qwen3.8 的 tokenizer）、完整 27B。
