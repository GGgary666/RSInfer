# MTP 投机解码：在 4090 D 的引擎上开发和验证（2026-11-13）

- 记录人：Claude Opus 5.5（Claude Code）。
- 做的事：照 SGLang `e838360` 搬 MTP 投机解码（[../specs/mtp.md](../specs/mtp.md)）。调度器和 worker 的逻辑先在 CPU 测试里写对，再上 GPU 验证、换真 kernel、录 graph。
- 机器：RTX 4090 D（24 GB）。Qwen3.8-27B 的真实结构截到 4 层，`deploy/agxinfer.py checkpoint --layers 4 --dtype bf16 --seed 1` 生成，现在带完整的随机 MTP 层（8.4 GiB）。
- 比较的办法：同样 4 个提示词、贪心、12 个 token 的文本（`ask.py`），单流 128 token 的速度（`speed.py`）。

## 结论

1. **投机解码能在跑着的引擎上打开**：`apply --config '{"spec.algorithm": "eagle"}'`，3.9 秒提交。调度宿主重载 `core.sched`、`core.state.alloc`，worker 重载 `models.qwen38`、`state.gdn_slots`、`layers.gdn`、`executors.pytorch`（和它们的使用者），holder 按新的预算重新分了 KV 页和 GDN 槽。
2. **GPU 上找到一个 CPU 测试没覆盖的 bug**：开重叠时，一行的第一个 decode 在它的 prefill 还在路上时排进来，计划里的位置是调度器知道的下限 0，executor 把它当成新行清掉了 GDN 状态。热改修复以后，开不开投机的文本相同。CPU 测试补了一个能复现它的版本（worker 慢一点、请求一个一个发）。
3. **先分清数值和 bug**：一开始 bf16、fp32 下开投机的文本都和不开不同，看起来像数值问题（verify 按 extend 算，和 decode 的 kernel 不同）。两边都换成参照 kernel、fp32 再比：不开重叠相同，开重叠不同，问题就落在重叠上了。
4. **4 层的模型上投机更慢，这是带宽决定的**：随机 MTP 几乎全拒，每一步读几遍 248k 词表的 LM head。Triton 的验证 kernel 和验证步的 graph 都没让它变快（设备时间本来就占满）。

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
| 验证步录 graph（`--cuda-graph auto`） | 三次冷启动：第一次启动时 1 lane 的 graph 对不上 eager；第二次 FlashInfer 的 buffer 在 inference mode 里建，之后 plan 写不进去；第三次通过 | 文本和 eager 验证相同，55.9 token/s；graph 确实在用（又热改进一个打印查过，再 apply 回去） |

一共冷启动约 14 次、热改 8 次。冷启动多，是因为比的就是两种配置各自冷启动的结果，以及 graph 只在 `core.plan` 加载时录、录不成引擎就起不来。

## 体会

- **CPU 测试抓不到依赖 GPU 异步的问题**：CPU 上 `submit` 返回时这一步已经算完了，调度器排下一步时上一步的结果多半已经回来，"位置是下限"的那条路走不到。要在 CPU 上复现，得让 worker 慢下来，并且让一行的 decode 紧跟着它自己的 prefill。
- **随机 MTP 权重测不了接受**：几乎每一步只接受 1 个，提交、草稿扩展取的行这些都走不到。用预言草稿（已知的贪心 token，定点出错），并且断言每一步接受的长度，才能抓到草稿那一侧的错（只比输出抓不到，验证会把错的草稿拒掉）。
- **在 4 层的模型上量投机的性能意义不大**：LM head 和 MTP 层占了大头，要在完整模型上量。
