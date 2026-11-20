# 把前缀缓存热改到一个跑着的引擎上，再在它上面验证（2026-11-13，RTX 4090 D）

记录人：Claude Opus 5.5（Claude Code）。对应的工作是 #59（分支 `dev/prefix-cache`，叠在 #58 上），设计和实现见 [../specs/prefix-cache.md](../specs/prefix-cache.md)。

前缀缓存先在 CPU 上写完、测完（§7.3 的测试）。这份记录写的是之后上 GPU 的部分：从 master 的代码起一个引擎，把 #59 热改上去，在引擎上看命中、对输出、量性能，最后停掉引擎跑 GPU 测试。

## 结论

1. **整个前缀缓存是一次 apply 热改上去的。** 新加了一个单元的模块、改了 contracts、调度器、账本、worker 里的 GDN 层和状态池、两个 kernel 单元，提交只用了 2.4 秒：准备 0.31 秒，暂停 0.015 秒，执行 0.63 秒（重建 GDN 状态池，重录 graph）。不改框架本体，所以不用重启。
2. **fp32 下，命中和不命中给出逐字相同的文本**，首 token 时间减半（共享约 600 token 的前缀：47 对 99 ms）。缓存的开关也是热改：`apply --config '{"alloc.radix_cache": false}'`。
3. **bf16 下，共享前缀的请求有一个和不开缓存时不一样。** 原因是 chunk 中间存下的状态取自 chunk kernel 的 `h`，`h` 是 bf16 的。SGLang 也是这样：它的 Triton GDN kernel 没有 fp32 快照，只有 KDA 的 kernel 有。照"和上游一样的不修"，没改。GPU 测试量了大小：
   - 存下的状态离单独算到那里的 2.4%，错一个 token 的对照差 130%；
   - 恢复后再算两个 token，状态离整个提示词算出的 2%，什么都不恢复的对照差 110%。
4. **性能**：
   - 共享约 3000 token 系统提示的请求，命中时首 token 27–35 ms，不开缓存 102–122 ms；
   - 不命中的 decode 吞吐不变（1 并发 169.2 对 169.2 token/s，8 并发 1272.5 对 1270.4）。graph 里那个跟踪 kernel 每步都发，没到间隔的步里它读一遍 mask 就退出。
5. **引擎重启了三次**，两次和前缀缓存无关：
   - 第一次：dummy 权重每个 token 都一样，对不了输出，换成随机 checkpoint 冷启动；
   - 第二次：热改成 fp32 时重读权重 OOM，撤回没得到确认，引擎要求恢复。写了报告 [../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md](../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md)；
   - 第三次：从 fp32 改回 bf16 同样会撞上这个 OOM，就直接冷启动了。
6. **在引擎上验证时发现测试本身不够严**：随机权重下 GDN 状态几十个 token 就淡掉了，恢复以后算 36 个 token 再比，没恢复也一样对得上。CPU 和 GPU 的恢复测试都改成恢复后只算几个 token，并带上"什么都不恢复"的对照。端到端测试也加了一个只剩 2 个 token 的请求：调度器不发恢复的拷贝，这条就失败。

## 过程

引擎的源码在 scratchpad 里单独的工作区（master 的 worktree），不是仓库的工作目录。热改就是把 #59 的 `engine/` 拷进这个工作区再 apply；改动都在 #59 的 worktree 里。

| 做了什么 | 计划 | 结果 |
|---|---|---|
| master `80ee578`，4 层 dummy 权重，`--cuda-graph auto --overlap --max-rows 8 --max-context 4096 --token-budget 2048`，冷启动 | | 输出每个 token 都是 `.`，对不了 |
| 停掉，`rsinfer checkpoint --layers 4 --dtype bf16 --seed 1`（14 秒，7.6 GiB），同样参数冷启动 | | 身份 `200ef73d5d49b01c`。基线：A、B、A2（同 A）、short 四个请求，A 和 A2 文本相同 |
| 把 #59 的 `engine/` 拷进工作区，`check`、`apply --dry-run`、`apply` | 调度宿主重载 `core.sched`、`core.state.alloc`、`sched.reqs`；worker 重载 `core.plan`、`core.worker`、`executors.pytorch`、`kernels.reference`、`kernels.triton`、`layers.attn`、`layers.gdn`、`state.gdn_slots`、`worker.batch`；前端只排空 | 提交，2.4 秒；调度器日志：`prefix cache on, 656 gdn.slots for at most 131 requests at once`。同样四个请求：A、A2、short 和基线相同，A2 首 token 32 → 22 ms；B 不同（见结论 3） |
| `apply --config '{"model.dtype": "f32", "state.gdn.conv_dtype": "f32"}'` | 重载 `core.state.weights`、`models.qwen38`、`layers.attn`、`layers.gdn`、`state.gdn_slots` | **失败，要求恢复**：重读权重时 OOM，撤回没得到确认。写了 framework-bugs 报告 |
| 停掉，`--dtype f32 --conv-dtype f32` 冷启动（#59 的代码） | | 250 个 GDN 槽，最多 50 个请求。四个请求开着缓存跑一遍 |
| `apply --config '{"alloc.radix_cache": false}'` | 重载 `core.state.alloc` | 提交，准备 0.28 秒、暂停 0.014 秒、执行 0.24 秒。同样四个请求：**四个文本都和开缓存时相同**；B 和 A2 的首 token 47.9 / 47.3 ms（开）对 99.3 / 102.0 ms（关） |
| 停掉，bf16 冷启动 | | 共享约 3000 token 系统提示的 8 个请求一个接一个发，首 token 依次是 228、27、29、28、29、33、33、35 ms；`tests/gpu/bench_serving.py`（不同的提示，256 个输出 token）1 并发 169.24 token/s、8 并发 1272.5 |
| `apply --config '{"alloc.radix_cache": false}'` | 同上 | 提交。同样的负载：首 token 122、112、110、108、103、104、102、107 ms；吞吐 169.24 / 1270.4 |
| 读 worker 日志里 `core.plan` 那一行 | | decode 和 graph 都用 `triton.gdn.track_states@1`；graph 留下的显存 65,011,712 字节，和 master 一样 |
| 停掉，跑 GPU 测试 | | 见下 |

GPU 测试（一次一个文件，`QWEN38_MODEL` 用上面那个 checkpoint）：9 个文件 53 项全部通过。

- 新的 `test_prefix_cache` 3 项。
- `test_real_kernels` 26 项。它列着 decode 用哪些 kernel，加上了 `gdn.track_states@1`。
- 其余照旧：`test_cuda_graph` 9、`test_cuda_graph_adversarial` 3、`test_cuda_graph_apply` 4、`test_qwen38_partial` 4、`test_qwen38_overlap` 1、`test_state_rebuild` 1、`test_qwen38_serve` 2。最后这个起整个引擎经 HTTP 对照 HF，现在缓存默认开着。

## 好用的地方

- **一次 apply 换掉一大块功能**，计划和实际做的一致：contracts 改了，用到它的单元（包括没改代码的 `sched.reqs`）都在计划里；状态单元有 `drain_on_reload`，在空引擎上重建。前端只排空，不重载。
- **用配置开关做对照**：`alloc.radix_cache` 是 startup 键，改它只重载账本，一次 apply 不到一秒。同一个引擎、同一份权重，开和关各跑一遍，比两次冷启动省事，也排除了"两次启动不一样"。
- **fp32 是正确性的标尺**：bf16 下一个请求不一样，换成 fp32 再比，四个相同，马上分清了是舍入还是 bug。

## 别扭的地方

- **换权重 dtype 的热改在小卡上走不通**：重读权重时旧的还留着备撤回，24 GB 放不下两份；更糟的是撤回没得到确认，引擎要求恢复（见上面的报告）。所以 dtype 的对照是靠冷启动做的。
- **对输出需要真正在变的权重**：dummy 权重每个 token 都一样，什么都对不出来；随机 checkpoint 才行。但随机权重让 GDN 的记忆很短，"恢复了没有"从输出上看不出来，要直接比状态，还要有对照。这在 CPU 测试里也藏着，是上 GPU 时才看出来的。
- **HTTP 接口只收文本提示**，没法把"上一个请求的提示加输出"按 token 原样送回去，decode 中途存下的状态在引擎上用不上。这部分改在 GPU 测试里用 token 直接测：graph 回放里存下的状态和 eager 的相同，没到间隔的槽一位不动。

## 没覆盖的

- SM100 上 64 层模型的效果和性能，以及和 SGLang 同配置、共享系统提示的负载对比：这台机器跑不了。
- A/B/C。
