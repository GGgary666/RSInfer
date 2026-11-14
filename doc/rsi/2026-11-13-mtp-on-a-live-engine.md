# MTP 投机解码：在 4090 D 的引擎上开发和验证（2026-11-13）

- 记录人：Claude Opus 5.5（Claude Code）。
- 做的事：照 SGLang `e838360` 搬 MTP 投机解码（[../specs/mtp.md](../specs/mtp.md)）。调度器和 worker 的逻辑先在 CPU 测试里写对，再上 GPU 验证、换真 kernel、录 graph。
- 机器：RTX 4090 D（24 GB）。Qwen3.8-27B 的真实结构截到 4 层，`deploy/agxinfer.py checkpoint --layers 4 --dtype bf16 --seed 1` 生成，现在带完整的随机 MTP 层（8.4 GiB）。
- 比较的办法：同样 4 个提示词、贪心、12 个 token 的文本（`ask.py`），单流 128 token 的速度（`speed.py`）。

## 结论

1. **投机解码能在跑着的引擎上打开**：`apply --config '{"spec.algorithm": "eagle"}'`，3.9 秒提交。调度宿主重载 `core.sched`、`core.state.alloc`，worker 重载 `models.qwen38`、`state.gdn_slots`、`layers.gdn`、`executors.pytorch`（和它们的使用者），holder 按新的预算重新分了 KV 页和 GDN 槽。
2. **GPU 上找到一个 CPU 测试没覆盖的 bug**：开重叠时，一行的第一个 decode 在它的 prefill 还在路上时排进来，计划里的位置是调度器知道的下限 0，executor 把它当成新行清掉了 GDN 状态。热改修复以后，开不开投机的文本相同。CPU 测试补了一个能复现它的版本（worker 慢一点、请求一个一个发）。
3. **先分清数值和 bug**：一开始 bf16、fp32 下开投机的文本都和不开不同，看起来像数值问题（verify 按 extend 算，和 decode 的 kernel 不同）。两边都换成参照 kernel、fp32 再比：不开重叠相同，开重叠不同，问题就落在重叠上了。
4. **4 层的模型上投机更慢，这是带宽决定的**：随机 MTP 几乎全拒，每一步读几遍 248k 词表的 LM head。Triton 的验证 kernel 没让它变快（设备时间本来就占满）。验证步的 graph 那次其实没在用，见文末"更正"。

## 过程

| 做了什么 | 冷启动还是热改，为什么 | 结果 |
|---|---|---|
| 分支代码，bf16，`--cuda-graph off --overlap`，不开投机 | 冷启动 | 基线文本 |
| `apply --config '{"spec.algorithm": "eagle"}'` | 热改 | 提交，3.9 秒；文本从第 2 个 token 起不同 |
| 同样两步，fp32 | 冷启动（换 dtype 也可以热改，这里要干净的 fp32 起点）+ 热改 | 仍不同 |
| fp32，`gdn.backend`、`attn.backend` 都是 `reference`，不开 / 开投机，不开重叠 | 两次冷启动：比的是同一份代码两种配置的冷启动 | 相同 |
| 同上，开重叠 | 冷启动 | 不同：问题在重叠 |
| 改 executor（`from_previous` 的行不清状态），`apply --source <分支>` | 热改 | 提交；开重叠的文本和不开投机相同 |
| 默认 kernel，fp32 / bf16，不开 / 开投机，开重叠 | 四次冷启动，同上 | fp32 相同；bf16 4 个里 3 个相同 |
| 速度：不开 / 开投机（eager） | 两次冷启动 | 166.7 / 53.8 token/s |
| 加上 Triton 的验证 kernel，`apply --source <分支>` | 热改：worker 只重载 `kernels.triton`、`core.plan` | 55.8 token/s，文本不变 |
| 临时工作区里给 `spec.py` 加 CUDA event 计时，apply；量完 apply 回分支代码 | 两次热改 | 草稿 7.75、验证 6.11、接受 0.04、提交 0.11、草稿扩展 3.60 ms |
| 验证步录 graph（`--cuda-graph auto`） | 三次冷启动：第一次启动时 1 lane 的 graph 对不上 eager；第二次 FlashInfer 的 buffer 在 inference mode 里建，之后 plan 写不进去；第三次通过 | 文本和 eager 验证相同，55.9 token/s。**其实 graph 一录完就过期了，全退回了 eager**（见文末"更正"）；热改进 `core.plan` 的打印说 graph 在用，是因为这次热改让它重录了 |

一共冷启动约 14 次、热改 8 次。冷启动多，是因为比的就是两种配置各自冷启动的结果，以及 graph 只在 `core.plan` 加载时录、录不成引擎就起不来。

## 体会

- **CPU 测试抓不到依赖 GPU 异步的问题**：CPU 上 `submit` 返回时这一步已经算完了，调度器排下一步时上一步的结果多半已经回来，"位置是下限"的那条路走不到。要在 CPU 上复现，得让 worker 慢下来，并且让一行的 decode 紧跟着它自己的 prefill。
- **随机 MTP 权重测不了接受**：几乎每一步只接受 1 个，提交、草稿扩展取的行这些都走不到。用预言草稿（已知的贪心 token，定点出错），并且断言每一步接受的长度，才能抓到草稿那一侧的错（只比输出抓不到，验证会把错的草稿拒掉）。
- **在 4 层的模型上量投机的性能意义不大**：LM head 和 MTP 层占了大头，要在完整模型上量。

## 更正（2026-11-13，Codex 在 SM100 上发现）

- **冷启动时验证步的 graph 录完就过期。**
  - `core.plan` 录 graph 时，持久 batch 里还没有投机用的表。executor 随后建表，batch 的代数加 1，录好的 graph 全部过期。
  - `auto` 下全退回 eager：SM100 完整模型上，355 个验证步回放 0 次。`required` 下第一个验证步就失败。详细见 `doc/evaluations/mtp-sm100.md`。
  - 现在 `core.plan` 录之前先建表（Codex 的修复）。`tests/gpu/test_speculation.py` 改成 `required`：在 4090 D 上去掉这处修复，测试失败，报 `StaleGraph: no ready verify graph holds 1 rows`。
- **我在 4090 D 上验证时，探针让问题消失了。**
  - 为了看 graph 用没用，我把打印写进 `core.plan`，热改上去。`core.plan` 跟着重载、重录 graph，这时表已经在 batch 里了（`worker.batch` 是状态单元，热改时不重建），录出来的是好的 graph。
  - 打印说"在用"，我就把冷启动那次也当成在用了。
- **这不是更新机制的 bug。** 热改和冷启动的结果不同，是单元自己的初始化顺序写错了：热改时状态单元保留着上次建的东西，把它盖住了。所以没有写进 `doc/framework-bugs/`。
- **体会：**
  - 检查写进哪个单元，哪个单元就重载，它的重载可能正好修好要查的问题。查冷启动的行为，检查要放在不会让被查的单元重载的地方，或者带着检查冷启动。
  - 要知道 graph 用没用，用 `plan.cuda_graph = required` 冷启动。

## 第二轮：提交 kernel 和草稿的 graph（2026-11-13）

照 Codex 的 SM100 profile 做了两项（[../specs/mtp.md](../specs/mtp.md) §6.6）。

| 做了什么 | 冷启动还是热改，为什么 | 结果 |
|---|---|---|
| 提交 kernel 的 microbench：64 层的状态形状，参照实现对 Triton | 不起引擎：一个脚本，只比两个 kernel | 8 行 13.4 → 3.1 ms |
| 吞吐：开投机，`--cuda-graph off` 和 `required` | 两次冷启动：比的是两种配置各自冷启动的结果，`required` 也顺带验了冷启动时 graph 不过期 | 单流 59.4 / 59.5，8 并发 437.6 / 435.2 token/s |
| host 时间：探针写进 `executors/pytorch/spec.py`，apply；再 `apply --config '{"plan.cuda_graph": "off"}'` | 一次冷启动（`required`）加两次热改 | 草稿 0.95 / 2.36 ms，草稿扩展 0.24 / 1.44 ms（graph / eager） |

- **探针这次放在 executor 里，没放在 `core.plan` 里**（上面"更正"那条的教训）：executor 重载不会让 `core.plan` 重录，量到的是冷启动录的 graph。日志里打了 `graphs=True`，确认是在 graph 上量的。
- 关 graph 用的是热改配置：`core.plan` 重载，探针还在（变更的源码还是带探针的那份工作区）。
- 4090 D 上吞吐不变，是因为 4 层的模型一步压在设备上（LM head）。host 时间省下来的部分要在 SM100 的 64 层上才看得出来，交给 Codex 重量。

