# 把 `model.dtype` 从 bf16 热改成 f32，重读权重时 CUDA OOM，撤回没得到确认，引擎要求恢复

- 日期：2026-11-13
- 报告人：Claude Opus 5.5（Claude Code），在 4090 D 上验证 #59（前缀缓存）时碰到，见 [../rsi/2026-11-13-prefix-cache-on-a-live-engine.md](../rsi/2026-11-13-prefix-cache-on-a-live-engine.md)
- 状态：已修（2026-11-13，PR #60，见文末「原因」「修复」）。是三个问题：权重存储申请前不把缓存还回去（撤回因此放不下，单元一侧）；状态存储不跟着 holder 重新定的数量走（f32 因此放不下，单元一侧）；撤回失败时诊断仍说"旧单元在服务"（框架本体，`unit.runtime`）。
- 引擎：
  - RTX 4090 D（24 GB）。Qwen3.8-27B 的真实结构截到 4 层，随机权重的 checkpoint：`deploy/agxinfer.py checkpoint --layers 4 --dtype bf16 --seed 1`，权重 7.57 GiB。
  - 起法：`deploy/agxinfer.py serve <ckpt> --tokenizer <Qwen3.6-27B> --cuda-graph auto --overlap --max-rows 8 --max-context 4096 --token-budget 2048`。代码是 master `80ee578`，之后热改上了 #59 的树。
  - 变更前身份 `200ef73d5d49b01c`，引擎是空的。holder 给状态分了 12.41 GiB（GDN 656 槽，KV 107,020 页）。
- 变更：

  ```sh
  python -m core.control.client --control <地址> apply --config '{"model.dtype": "f32", "state.gdn.conv_dtype": "f32"}'
  ```

  `--dry-run` 的计划：worker 重载 `core.state.weights`、`models.qwen38`、`layers.attn`、`layers.gdn`、`state.gdn_slots`，不要求重启。
- 现象：`status: failed`，`recovery_required: true`。耗时：check+plan 1.10 秒、prepare 0.19 秒、pause 0.011 秒、abort 0.32 秒。诊断（控制面 `events.sqlite` 的 `changes.diagnostics_json`）：

  ```text
  models.qwen38 | load | unit.not_active | is FAILED after reload: OutOfMemoryError: CUDA out of memory.
    Tried to allocate 4.74 GiB. GPU 0 has a total capacity of 23.52 GiB of which 3.62 GiB is free. Including
    non-PyTorch memory, this process has 19.89 GiB memory in use. Of the allocated memory 17.39 GiB is
    allocated by PyTorch, and 2.07 GiB is reserved by PyTorch but unallocated.
  core.plan / core.worker / executors.pytorch | load | unit.not_active | is PENDING after reload, waiting for model@1 / executor@1 / plan@1
  | rollback | unit.rolled_back | reload rolled back; the previous units are serving
  | rollback | escalate.restart_process | units did not come back: ['core.plan', 'core.worker', 'executors.pytorch', 'models.qwen38']
  host-a/worker/0 | abort | change.abort_unacknowledged | rollback not confirmed: ... (hint: restart the process)
  | abort | change.recovery_required | rollback or process stop is unconfirmed; restart the engine
  ```

  - 4.74 GiB 正好是 f32 的词表矩阵（248,320 × 5,120 × 4 字节）。旧的 bf16 权重和状态池都还在：单元运行时留着旧实例，备着撤回。**（后来查明这句不对：旧实例已经卸掉、显存已经释放，见「原因」。）**
  - 之后 `status` 里 worker 的单元全是 ACTIVE，身份还是旧的。这时没再发请求试，因为要求恢复以后 `apply` 都被拒。**（后来查明这是过期的记录：`models.qwen38` 其实是 FAILED，请求发过去没有回应，见「原因」第 4 条。）**
- 期望：
  - 放不下就失败，这没问题：要同时放得下新旧两份权重，24 GB 的卡上放不下。**（后来查明：不用同时放两份，同样的配置 f32 冷启动放得下，热改也应当放得下，见「原因」第 2 条。）**
  - 但撤回应当是干净的，像 [2026-11-10-gdn-ssm-dtype-rebuild-oom.md](2026-11-10-gdn-ssm-dtype-rebuild-oom.md) 那次一样：`aborted`，旧版本照常服务。而不是一面说"已撤回，旧单元在服务"，一面说"单元没回来"，最后要求恢复。
  - 两条诊断互相矛盾，说明撤回在确认哪些单元回来了这一步上判断错了，或者真有单元没回来。没查是哪一种。
- 复现：同样的引擎，同样的 `apply --config`。在一张放不下两份权重的卡上应当每次都复现；没有写成测试。
- 绕过：停掉引擎，用 `--dtype f32 --conv-dtype f32` 冷启动，接着做别的验证。之后没有再试这个热改。

## 原因（2026-11-13，在 4090 D 上复现后查明）

同样的引擎（4 层 bf16，dummy 权重或随机 checkpoint，参数同上）、同样的 `apply`，每次都复现。在临时工作区的 `unit.runtime` 里加打印，记下每一步的显存（PyTorch 实际在用的、它缓存着的，以及循环引用里没回收的张量）：

| 时刻 | 在用 | PyTorch 占着 | 说明 |
|---|---:|---:|---|
| 旧单元卸掉以后 | 6.98 GiB | 20.46 GiB | 只剩没重载的 KV 池等；旧权重和旧 GDN 池都释放了，循环引用里没有张量 |
| 新单元失败时 | 17.38 GiB | 19.45 GiB | KV 6.53 + 新的 f32 GDN 池约 6.1 + 第一个 f32 词表矩阵 4.74；第二个词表矩阵放不下 |
| 撤回卸掉新实例以后 | 6.98 GiB | 19.45 GiB | 回到卸载后的水平，释放的块全留在 PyTorch 的缓存里 |
| 撤回重新加载旧单元以后 | 14.55 GiB | 19.45 GiB | 旧的 bf16 权重放进去了；GDN 池要 5.78 GiB，OOM |

1. **报告里"旧实例留着备撤回"不对。** 单元运行时撤回时是把旧单元重新加载成新实例，不留旧实例；卸掉的实例的显存也真的释放了。
2. **f32 放不下，是因为状态存储不跟着 holder 走。** 模型每次加载都让 holder 按权重大小重新定状态的数量（`engine/models/qwen38/unit.py` 的 `start`）。可是 `state.kv_paged` 和 `state.gdn_slots` 只在第一次声明时取 holder 的数，之后一直用它。这次变更不重载 KV 存储（它的 dtype 是 `auto`，跟着 `model.dtype` 变），它就在原地按 bf16 时的 107,020 页、用 f32 重新申请，比 holder 给的预算多一倍多（第二次复现时就失败在这块 6.53 GiB 上）。同样配置用 f32 冷启动，holder 给 250 个槽、20,934 页，放得下。2026-11-10 那份报告最后"另外观察到"的就是这件事。
3. **撤回失败是 PyTorch 缓存的碎片。** 撤回时旧的 bf16 权重先申请，从失败的那次释放的大块（f32 权重、f32 GDN 池）里按 best-fit 切出来，把这些块钉住了；接着 GDN 池要一整块 5.78 GiB，缓存里没有，设备上只剩 3.63 GiB，于是 OOM（"4.78 GiB reserved by PyTorch but unallocated"）。总量其实够。这和 [2026-11-10-gdn-ssm-dtype-rebuild-oom.md](2026-11-10-gdn-ssm-dtype-rebuild-oom.md) 是同一个机制，#46 只让状态存储先 `empty_cache` 再申请，权重存储没有，而撤回时偏偏是权重先申请。`models.qwen38` 没回来，用它的 `core.plan`、`executors.pytorch`、`core.worker` 也起不来。之后要求恢复是对的：撤回真的失败了。
4. **两处诊断误导了人：**
   - `unit.runtime` 的 `_rollback` 不管有没有单元没回来，都先报一句 "reload rolled back; the previous units are serving"，再报 "units did not come back"，也不说为什么没回来。
   - `status` 里每个进程的单元，是它上次检入、确认提交或确认撤回时报的。撤回没得到确认，这份记录就不更新：`models.qwen38` 实际是 FAILED、第 3 代，`status` 还显示 ACTIVE、第 1 代。`introspect` 是现查各进程的，看得到真实状态。这一条这次不修（要改 `core.change` 的 coordinator），`status` 里同时有 `recovery_required: true`。
5. 另外更正 [../rsi/2026-11-13-prefix-cache-on-a-live-engine.md](../rsi/2026-11-13-prefix-cache-on-a-live-engine.md) 里的一句："从 fp32 改回 bf16 同样会撞上这个 OOM"是当时没试就写的。试了，不修也能提交：权重变小，KV 池按原来的页数用 bf16 申请，比预算少。

## 修复（2026-11-13）

- **`core.state.weights`**（单元）：`WeightStore.declare` 先放掉要换的缓冲区、`torch.cuda.empty_cache()`，再申请，和两个状态存储一样。
- **`state.kv_paged`、`state.gdn_slots`**（单元）：数量没有显式配置（`state.kv.pages` / `state.gdn.slots` 为 0）时，每次声明都重新取 holder 的数；数变了，整个池重新申请。
- **`unit.runtime`**（框架本体，`engine/unit/runtime/runtime.py` 的 `_rollback`）：有单元没回来时，只报一条错误，写明每个单元的状态和原因（"reload rolled back, but units did not come back: models.qwen38 is FAILED: OutOfMemoryError ..."），不再说旧单元在服务。单元一侧做不到：这句话是单元运行时自己说的。
- 测试：
  - `tests/gpu/test_state_rebuild.py::test_old_weights_and_pool_fit_again_after_a_larger_attempt_failed`：按设备空闲显存定大小，旧权重和旧池卸掉、更大的一次失败、再装回旧的。拿掉权重存储的修复，这条失败在池子上（要 10.41 GiB，设备只剩 8.09 GiB）。
  - `tests/units/executors/test_config.py::test_a_worker_rebuilt_for_new_state_config_generates_as_one_started_with_it[model.dtype-bf16-...]`：数量由 holder 定时，原地改 `model.dtype` 以后的容量和输出与冷启动相同。拿掉状态存储的修复，容量对不上。
  - `tests/framework/unit_runtime/test_reload.py::test_a_rollback_that_cannot_bring_a_unit_back_says_which_and_why_and_not_that_the_old_units_serve`：旧单元在撤回时起不来。拿掉 `_rollback` 的修复，这条失败。
- 在引擎上（4090 D，4 层随机 checkpoint，过程见 [../rsi/2026-11-13-weight-swap-on-a-live-engine.md](../rsi/2026-11-13-weight-swap-on-a-live-engine.md)）：
  - 两个单元的修复热改到 master 的引擎上，提交，输出不变；
  - 同样的 `apply --config '{"model.dtype": "f32", "state.gdn.conv_dtype": "f32"}'`：**提交**，holder 给 250 个槽、20,934 页，和 f32 冷启动相同，输出和 f32 冷启动逐字相同；改回 bf16 也提交，输出回到最初的；
  - 显式定死池子的大小（`state.kv.pages=107020`、`state.gdn.slots=656`）再改 f32，真的放不下：`aborted`，"reload rolled back; the previous units are serving"，不要求恢复，输出和之前相同。
