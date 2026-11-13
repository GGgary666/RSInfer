# 前缀缓存第二轮评审（Antigravity）的修复，在跑着的引擎上验证（2026-11-13，CPU）

记录人：Claude Opus 5.5（Claude Code）。对应 #59，Antigravity 的评审基于 `d1635fc`。上一轮（Codex）见 [2026-11-13-prefix-cache-review-fixes-on-a-live-engine.md](2026-11-13-prefix-cache-review-fixes-on-a-live-engine.md)，设计和修改记在 [../specs/prefix-cache.md](../specs/prefix-cache.md) §7.2、§7.3。

## 结论

五条都成立，都修了：

1. **换 worker 时，刚从缓存接纳的请求接着旧 worker 的状态算**（P1）。修法：`ledger@1` 加 `restart`。`_lose` 里，算过 token 却没有一步经旧连接发出去的请求，起点只能来自缓存，让它从 0 重算；没有空页时它结束。没修时，夹具 worker 发现请求 2 从 64 接着算、页是空的，这一步失败；真 worker 上则是静默算错，Antigravity 的复现就是这样。修完请求 2 先被新 worker 拒绝（从 64），再从 0 算，输出照常。
2. **页和 GDN chunk 互不整除时 worker 停机**（P1）。修法：照搬 SGLang 的两条检查：页和 64 一个整除另一个，`mamba_track_interval` 是页的整数倍。
3. **到不了一页的 chunk 也先接纳**，接纳时从缓存淘汰（P2）。修法：先用此刻的命中算这一块，到不了一页就不接纳，和 SGLang `PrefillAdder` 的顺序一样。
4. **`flush` 以后还给出之前记下的命中**（P2）。修法：`flush` 清掉记下的命中。另外 `tests/units/worker/test_prefix_cache.py` 查命中时都用请求 id 0，后两个直接拿了第一个的结果；改成各用各的 id。Antigravity 还建议截断 `n`：这一块不超过提示词剩下的长度。这一条也做了（`_chunk`）。
5. **`RadixTree` 自己不初始化 `freed_states`**（P3）。修法：在它的 `__init__` 里初始化。

热改本身：改了 `core/contracts/ledger.py`。计划只重载调度宿主的 `core.sched` 和 `core.state.alloc`，worker 不动，一次提交。

## 过程

小随机模型（4 层，fp32，`agxinfer checkpoint --seed 3`），`--device cpu --max-rows 8 --cpu-bytes 16MiB`，`state.gdn.slots = 20`。

| 做了什么 | 计划 | 结果 |
|---|---|---|
| 冷启动 `d1635fc`，页 4，预算 128，`state.kv.pages = 160` | | 100 token 的请求两遍相同 |
| `apply --config '{"state.kv.page_size": 96, "sched.token_budget": 192}'` | 重载 worker 的 `state.kv_paged` 和账本 | `aborted`：账本的 `alloc.states` 写着页 4，页大小只能冷启动时定。这里报的是原有的页大小检查，不是这次的 bug |
| 把修好的四个文件（`contracts/ledger.py`、`sched/loop.py`、`state/alloc/radix.py`、`state/alloc/unit.py`）拷进工作区，`check`、`apply --dry-run`、`apply` | 只重载调度宿主的 `core.sched`、`core.state.alloc` | 提交 |
| 100（两遍）、66、80、130 token 的请求，前缀相同；然后 `apply --config '{"alloc.radix_cache": false}'` 再发一遍，最后开回来 | 只重载账本和调度器 | 开和关的文本逐字相同。这次没在引擎里加检查，命中次数没数（命中的逻辑由账本测试覆盖） |
| 停掉，用 `d1635fc` 以页 96、预算 192、`state.kv.pages = 40` 冷启动 | | 90 token 的请求正常；**100 token 的请求停掉了引擎**：`row 0 keeps its state at 96 ... not its end nor a multiple of 64`，之后的请求都报这个错 |
| 在这个引擎上 apply 修好的代码 | 同上 | `aborted`：`state.kv.page_size = 96: ... must divide the other (as SGLang's)`，撤回 |
| 停掉，用修好的代码以页 96 冷启动 | | 起不来，`serve` 退出码 1，日志里是同一句 |

第 1、3 条在引擎上不好造：
- 第 1 条要换 worker，而且新 worker 报 `Capacity` 之前，调度器得先给它发一步；
- 第 3 条要看树里还剩几个状态。

这两条都用调度器测试台验证（夹具 worker），见 §7.3。

## 好用的地方

- **改 contracts 也是一次热改**：计划准确地只重载了 import 它的两个单元。
- **在引擎上复现第 2 条很直接**：冷启动、发一个请求，报的错和评审里写的逐字相同。修好的代码 apply 上去被撤回，诊断说清了原因，引擎照旧。

## 别扭的地方

- **页大小不能热改**：它写在 `alloc.states`（模型的 StateDecl）里，`deploy` 启动时算好。所以第 2 条的对照只能冷启动。
