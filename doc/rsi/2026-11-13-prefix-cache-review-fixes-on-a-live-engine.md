# 在跑着的引擎上复现并修掉 Codex 对前缀缓存的三条评审（2026-11-13，CPU）

记录人：Claude Opus 5.5（Claude Code）。对应 #59（分支 `dev/prefix-cache`），Codex 的评审基于 `eeeaccf`。设计和修改记在 [../specs/prefix-cache.md](../specs/prefix-cache.md) §7.2、§7.3。

## 结论

三条都成立，都修了：

1. **撤回的变更没丢缓存**（P1）。变更重建了 GDN 状态池，后面的单元加载失败、撤回时旧单元重新加载，池子是全零的；账本没重载，树还指着那些槽。在引擎上看到：撤回以后 4 个请求都命中 64，恢复的状态范数是 0（撤回前是 0.018–0.020），4 个输出里 2 个和冷跑不同。修法：排空的 `abort` 也 `flush`。修完同样的过程：撤回以后 4 个请求都从 0 开始，不恢复，输出和冷跑逐字相同。
2. **预算小于一页时请求一直挂着**（P2）。页 4、`sched.token_budget = 2`：旧代码提交了这个配置，66 token 的请求 5 秒没回来。修法：开着缓存时预算必须是页的整数倍（SGLang 的 `chunked_prefill_size % page_size == 0`），调度器加载时检查。修完同样的 apply 被撤回，诊断里写着原因，引擎照常用 128 服务。
3. **记下的命中一直涨**（P2）。修法：接纳和结束时删掉，换新树时全清。修完 8 个并发客户端发 300 个短请求，每次接纳时字典里只有 0–5 条（最多 4 个在跑，其余在等）。顺带修了一个同源的问题：新树的版本号从 0 重新数，旧树上记下的命中会被当成新树的（账本测试里能构造出来：应为 192，旧代码给 0）。反过来时（记下的比实际短），照代码推，接纳以后这一块会超过提示词的长度；这种情况在引擎上没碰到。

## 过程

引擎的源码在 scratchpad 里单独的工作区，先放 `eeeaccf` 的代码。小随机模型（4 层，fp32，`agxinfer checkpoint --seed 3`），`--device cpu --page-size 4 --max-rows 8 --max-context 256 --token-budget 128`，`state.gdn.slots = 20`、`state.kv.pages = 160`。

| 做了什么 | 计划 | 结果 |
|---|---|---|
| 冷启动 `eeeaccf` | | 20 个 GDN 槽，最多 4 个请求 |
| 16 个随机 66 token 提示，各发两遍 | | 第二遍和第一遍相同，但一个都没命中：16 × 73 token 要约 290 页，只有 160 页，LRU 在它们回来之前就淘汰了 |
| 工作区里临时加两处检查：`state.gdn_slots.copy` 打印恢复的状态范数，账本打印每次接纳的前缀；apply | 重载 `state.gdn_slots`（及用它的单元），后一次重载 `core.state.alloc`、`core.sched` | 提交。临时检查不进仓库 |
| 改成 4 个提示：冷跑、再跑一遍、`apply --config '{"state.gdn.slots": 21, "plan.cuda_graph": "required"}'`、再跑一遍 | worker 重载 `state.gdn_slots`、`core.plan` | 再跑一遍时 4 个都命中 64，恢复范数 0.018–0.020；apply 在 CPU 上要求 graph 失败，`aborted`、`unit.rolled_back`；**撤回后 4 个都命中 64，恢复范数全是 0，2 个输出和冷跑不同** |
| `apply --config '{"sched.token_budget": 2}'`，发一个请求 | 重载 `core.sched` | 提交；**请求 5 秒没回来**。`--force` 改回 128，恢复服务 |
| 把修好的 `drain.py`、`loop.py`、`radix.py` 拷进工作区（账本的检查重新加上），`apply --dry-run`、`apply` | 只重载调度宿主的 `core.sched`、`core.state.alloc`，worker 不动 | 提交 |
| 同样的 4 个提示和失败的 apply | | 冷跑和旧代码的冷跑逐字相同；再跑一遍命中、恢复范数同上；**撤回后 4 个都从 0 开始，没有恢复，输出和冷跑相同** |
| `apply --config '{"sched.token_budget": 2}'` | | `aborted`：`sched.token_budget = 2 must be a multiple of the prefix cache's page, state.kv.page_size = 4 ...`，撤回；请求照常 0.07 秒回来，配置还是 128 |
| 8 个并发客户端发 300 个两 token 的短请求 | | 每次接纳时字典 0–5 条 |
| 停掉引擎 | | |

## 好用的地方

- **在引擎里加临时检查很快**：两处打印，一次 apply 就上去，直接看到"命中了、恢复的是全零的槽"。光看输出会以为没这个 bug：这组随机权重下状态很小，16 个提示里一个输出都没变。
- **修复只重载调度宿主**：计划里 worker 不动，状态池和里面缓存的东西都留着。

## 别扭的地方

- **在 CPU 上制造"重建以后失败"要绕一下**：靠的是 CPU 上 `plan.cuda_graph = required` 加载必然失败。
- **测试台的引擎加载失败会把 pytest 挂两分钟**：调度器拒绝配置时，`tests/units/scheduler/support.py` 的 `Engine` 不关自己的 zmq socket，垃圾回收时 `Context.term()` 一直等，直到单个测试的超时。改成加载失败时先关掉再抛。
