# 换权重和撤回的显存：在 4090 D 的引擎上查和修（2026-11-13）

- 记录人：Claude Opus 5.5（Claude Code）。
- 起因：[../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md](../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md)。那天验证前缀缓存时，把 `model.dtype` 从 bf16 热改成 f32 失败了，撤回也没成功，引擎要求恢复，dtype 的对照只好靠冷启动做。这次查清原因、修掉，再确认换权重在跑着的引擎上能做。
- 机器：RTX 4090 D（24 GB）。Qwen3.8-27B 的真实结构截到 4 层：查问题时用 dummy 权重，验证时用两份随机 checkpoint（`deploy/agxinfer.py checkpoint --layers 4 --dtype bf16 --seed 1`、`--seed 2`，各 7.6 GiB）。
- 起法都一样：`serve <ckpt> --tokenizer <Qwen3.6-27B> --cuda-graph auto --overlap --max-rows 8 --max-context 4096 --token-budget 2048`。每次比较都用同样四个提示词、贪心、12 个 token。

## 结论

1. **原来的报告有两处看错了。** 撤回时旧单元是重新加载成新实例，不是"留着旧实例备撤回"，旧权重也确实释放了。之后 `status` 里"全是 ACTIVE"是过期的记录，`models.qwen38` 其实是 FAILED，请求发过去没有回应。
2. **f32 放不下，是状态存储不跟着 holder 走。** 模型每次加载都让 holder 按新权重的大小重新定状态数量，可 KV 存储没被重载，还按 bf16 时的 107,020 页在原地用 f32 申请。改成每次声明都取 holder 的数以后，热改 f32 提交了，holder 给的数和输出都和 f32 冷启动相同。
3. **撤回失败是 PyTorch 缓存的碎片。** 撤回时旧权重先申请，从失败那次释放的大块里切出来，把它们钉住，旧的 GDN 池就放不下了。权重存储改成先 `empty_cache` 再申请（状态存储 2026-11-10 起就这样做）以后，真的放不下的变更干净地 `aborted`，旧版本照常服务。
4. **换权重在跑着的引擎上能做**：换 checkpoint（`weights.path`）、换 dtype（`model.dtype`），都是一次 `apply --config`，输出和用新配置冷启动逐字相同。`refit` 动作也在，但控制接口还没有请求它的入口。
5. **框架本体改了一处**：`unit.runtime` 的 `_rollback` 在有单元没回来时，不再说"旧单元在服务"，而是写明哪个单元、什么状态、为什么。`status` 显示过期单元这一条没改（要改 `core.change`），写进了 skill。

## 过程

### 查原因

| 做了什么 | 冷启动还是热改，为什么 | 结果 |
|---|---|---|
| 临时工作区（master 加 `# PROBE` 打印），4 层 bf16 dummy | 冷启动：打印加在 `unit.runtime` 的 `_reload`、`_rollback` 里，那是框架本体，热改不上去 | holder 给 656 个槽、107,020 页 |
| `apply --config '{"model.dtype": "f32", "state.gdn.conv_dtype": "f32"}'` | 热改 | 和报告一样：`failed`、要求恢复。打印见 framework-bugs 报告的表：旧单元卸掉以后只剩 6.98 GiB 在用；撤回时 GDN 池要 5.78 GiB，设备剩 3.63 GiB，PyTorch 缓存里还空着 4.78 GiB |
| `status` 和发一个请求 | | `status` 里 worker 的单元全是 ACTIVE、第 1 代；请求 60 秒没有回应 |
| 临时工作区的权重存储加上先 `empty_cache`，同样起、同样 apply | 冷启动：上一个引擎要求恢复，什么都 apply 不上去 | `aborted`，"previous units are serving"，请求有回应，`status` 不要求恢复，撤回后在用的显存回到启动时的 20.49 GiB。失败的位置变成了 6.53 GiB（KV 池），由此看出 KV 池是按旧页数用 f32 申请的 |
| f32 冷启动，再改 bf16，带修复和不带修复各一次 | 冷启动：要从 f32 开始 | 都提交。所以"从 f32 改回 bf16 也会 OOM"是没试就写的 |

### 验证修复

| 做了什么 | 计划 | 结果 |
|---|---|---|
| 临时工作区 = master，ckpt1 冷启动 | | 基线文本 |
| 把分支上两个状态存储和权重存储的改动（不含 `unit.runtime`）拷进工作区，apply | worker 重载 `core.state.weights`、`state.gdn_slots`、`state.kv_paged`（和用它们的单元） | 提交，文本和基线相同 |
| `model.dtype` 和 `state.gdn.conv_dtype` 改 f32 | 重载权重存储、模型、两个层、GDN 存储 | **提交**。holder：250 个槽、20,934 页，和 f32 冷启动相同；显存 21,870 MiB |
| 改回 bf16 | 同上 | 提交，文本回到基线 |
| `weights.path` 改成 ckpt2 | dry-run：worker 重载 `core.state.weights` | 提交，整个 apply 3.2 秒，文本变了 |
| `weights.path` 改成不存在的目录 | 同上 | `aborted`，诊断是 `FileNotFoundError: no *.safetensors`，"previous units are serving"，文本还是 ckpt2 的 |
| `weights.path` 改回 ckpt1 | 同上 | 提交，文本回到基线 |
| 停掉，分支（含 `unit.runtime` 的改动）冷启动 ckpt2 | 冷启动：要对照热改的结果；分支改了框架本体，本来也只能冷启动带进去 | 文本和热换成 ckpt2 的逐字相同 |
| 分支冷启动 ckpt1，f32 | 同上 | 文本和热改成 f32 的逐字相同 |
| 分支冷启动 ckpt1，bf16，`--set state.kv.pages=107020 --set state.gdn.slots=656`，再改 f32 | 池子定死了大小，f32 真的放不下 | `aborted`，"reload rolled back; the previous units are serving"，不要求恢复，worker 的单元是新的一代，文本和之前相同 |

中间误把改成 ckpt2 的配置又 apply 了一次，提交了，没有影响。

一共冷启动 8 次、热改 12 次（不算 dry-run）：查原因时 4 次和 4 次，验证时 4 次和 8 次。冷启动的原因都写在表里：查问题的打印在框架本体里；引擎要求恢复以后只能重启；从 f32 起步；和冷启动对照；分支改了框架本体。

### 测试

- CPU：`uv run pytest -n 8` 全过，pyright 0 个错误。新加的三条测试各自拿掉对应的修复都失败（见 framework-bugs 报告的「修复」）。
- GPU（4090 D，一次一个文件，`QWEN38_MODEL` 用 ckpt1）：9 个文件 54 项全过（`test_state_rebuild` 2、`test_prefix_cache` 3、`test_cuda_graph` 9、`test_cuda_graph_apply` 4、`test_cuda_graph_adversarial` 3、`test_real_kernels` 26、`test_qwen38_partial` 4、`test_qwen38_overlap` 1、`test_qwen38_serve` 2）。`test_prefix_cache` 改过的那条在 4 层上：存下的状态 1.5e-2（conv）、2.4e-2（temporal），错一个 token 1.4、1.3，是 10 倍门槛的五倍以上。

## 体会

- **先查清楚再写结论。** 原来的报告凭一条 OOM 的信息就写了"旧实例还留着"，又凭 `status` 写了"单元都回来了"，两条都错。在 `_rollback` 前后打出显存（在用、PyTorch 占着、循环引用里没回收的张量）以后，一次就看清楚了。办法写进了 skill「在引擎里做检查」。
- **`status` 不是现查的**，撤回没得到确认时尤其不能信，看 `introspect`。
- **OOM 的信息要看全**："reserved by PyTorch but unallocated" 加上设备空闲，比要的多，就是碎片，不是真的不够。
- **换 dtype 不用重启了**，以后对照 bf16 和 fp32 可以在同一个引擎上来回改。
