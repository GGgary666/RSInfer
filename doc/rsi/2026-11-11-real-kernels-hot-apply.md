# 真 kernel 迁移里热更新用了多少，和补做的一次整批热改（2026-11-11，RTX 4090 D）

记录人：Claude Fable 5.2（Claude Code）。对应的工作是 #51 的两批真 kernel（[../evaluations/real-kernels-4090d.md](../evaluations/real-kernels-4090d.md)、[../evaluations/real-kernels-fused.md](../evaluations/real-kernels-fused.md)）。

这份记录回答两件事：

- 这次工作里，我有没有照 AGENTS.md 说的"在跑着的引擎上热改"来开发；
- 这次最大的一个改动能不能热改，热改出来的引擎和冷启动的是不是一样。

## 结论

1. **没有照着做。** 两批工作的开发循环都是冷启动：pytest 和 profile 脚本在进程内搭 worker 栈，压测时冷启动引擎。热更新只在两轮收尾验证和一次配置变更里用过。
2. **用到的部分没有出过问题。** 试验之前 14 次 apply 全部提交；7 次起引擎里没有一次是热更新失败逼出来的。
3. **补做的试验：整个第二批可以热改，结果和冷启动一样。** 从第一批的树冷启动引擎，把第二批的树整个 apply 上去：3.1 秒提交，不重启任何进程，worker 重载 11 个单元。选的 kernel、graph 桶、输出的 token、吞吐、身份都和用新树冷启动的相同。
4. **带在途请求的三种情况都对**（拒绝、`--wait-empty`、`--force`），来回 9 次没有报错，显存不涨。
5. **真正挡着我一直用热更新的是两件事**：这张 24 GB 的卡放不下引擎和测试同时跑；大部分验证要进到 worker 进程里看张量。第二件事现在没有办法在跑着的引擎上做。

## 这次工作里实际用了多少

数字是从会话记录里数出来的，不含测试自己起的引擎。

起引擎 7 次（`deploy/qwen38.py serve`，4 层、bf16）：

| 第几次 | 为什么起 | 之后在它上面做了什么 |
|---:|---|---|
| 1 | 第一批收尾，验证热改 | apply 9 次：改 `kernels.triton` 再改回，连续重载 `kernels.flashinfer` 4 次，配置切 backend 3 次 |
| 2 | 和原生 SGLang 对比的压测 | apply 1 次：用配置把 graph 桶和 `sched.token_budget` 调大。压测时 worker OOM，引擎停止服务 |
| 3 | 上一个坏了以后重启，换了内存划分 | 压测 |
| 4 | 换回默认的内存划分重测 | 压测 |
| 5 | 换上 RMSNorm 的实验代码 | 压测 |
| 6 | 换上第二批的融合 kernel | 压测 |
| 7 | 第二批收尾，验证热改 | apply 4 次：backend 切到 `reference` 再切回，临时改 packed decode 的代码再改回 |

- 第 3 次是唯一一次因为故障重启。原因是我配错了：`sched.token_budget` 16384 加默认的 `holder.mem_fraction` 放不下一步 prefill 的瞬时显存。apply 接受了这个配置，这不是更新机制的 bug，但后果是引擎停了（见"碰到的事"）。
- **第 5、6 次本来可以直接 apply。** 我是停掉引擎、换代码、再冷启动的。下面的试验说明第 6 次那个改动热改没有问题。
- 第 4 次改的是启动时定的内存划分，我没有试 apply 会给出什么计划。
- 每两次之间引擎都被我停掉了，因为 GPU 测试、profile 脚本、microbench 和原生 SGLang 都要这张卡。

## 补做的试验：把整个第二批热改到第一批的引擎上

第二批是这次最大的改动：换掉三个 kernel definition，改 `core/contracts/plan.py` 里 `GDNBackend` 的接口，同时改模型、五个层单元、三个 kernel 单元和 `core.plan`。

### 做法

- 工作区是一个 git worktree，先停在 `f096075`（第一批）。在里面起引擎：

  ```sh
  python deploy/qwen38.py serve --model <4 层 bf16 checkpoint> --tokenizer <Qwen 的 tokenizer> \
      --cuda-graph auto --overlap --max-rows 32 --max-context 4096
  ```

- 把工作区切到 `a471a04`（第二批），然后 `check`、`apply --dry-run`、`apply --source <工作区>`。
- 对照：用 `a471a04` 冷启动一个引擎，同样的参数。
- 每个状态记四样：身份、worker 日志里最新的 `core.plan: {...}`、`nvidia-smi` 的显存、4 个 prompt 各 32 个贪心 token（一次一个请求）。

### 计划

`check` 没有诊断。dry-run 和实际执行的计划相同：

- 不重启引擎，不重启任何进程；
- worker 重载 11 个单元：`core.plan`、`executors.pytorch`、`kernels.flashinfer`、`kernels.reference`、`kernels.triton`、`layers.activation`、`layers.attn`、`layers.gdn`、`layers.norm`、`layers.rope`、`models.qwen38`；
- 前端和调度宿主不重载，只检查引擎是不是空的。

我对着两个提交的 diff 和 import 关系核过：10 个是代码变了的单元；`executors.pytorch` 自己没变，但它 import 了变了的 `core/contracts/plan.py`。没有漏的，也没有多的。权重存储、两个状态池、kernel 目录、持久 batch、worker 循环都没有动。

### 在第一个引擎上做的全部变更

| 变更 | 引擎里有请求吗 | 结果 | check+plan | drain | execute | commit |
|---|---|---|---:|---:|---:|---:|
| 启动（旧树） | | 提交 | 0.91 | | 10.01 | 0.37 |
| 新树，dry-run | 没有 | 只出计划 | | | | |
| 新树 | 没有 | 提交 | 0.97 | 0.00 | 0.97 | 0.40 |
| 旧树，不带参数 | 有 1 个在生成 | 拒绝 | 1.03 | | | |
| 旧树，`--wait-empty 120` | 同一个请求 | 提交 | 1.05 | 10.54 | 0.20 | 0.39 |
| 新树，`--force` | 有 1 个在生成 | 提交 | 0.99 | 0.02 | 0.26 | 0.39 |
| 旧树、新树来回 3 轮 | 没有 | 6 次都提交 | 0.95–1.03 | 0.00–0.01 | 旧 0.23，新 0.27–0.46 | 0.39–0.41 |

单位是秒。

- **第一次 apply 从发出到返回 3.1 秒。** execute 0.97 秒里包括重录 8 个 graph 桶；之后每次约 2 秒。冷启动光启动那次变更的 execute 就是 10–11 秒。
- 从头到尾是同一个 worker 进程，这个改动在它上面提交了 9 次。各进程的日志里没有 Traceback 或 Error。

### 热改出来的和冷启动的比

新树（`a471a04`）：

| | 热改（第 1 次） | 热改（`--force` 那次） | 热改（再来回 3 轮以后） | 冷启动 |
|---|---|---|---|---|
| 身份 | `c3191ac79f52cbd4` | 相同 | 相同 | 相同 |
| `core.plan` 的启动记录（各阶段选的 kernel、graph 桶） | 和冷启动逐项相同 | 相同 | 相同 | |
| graph 桶 | 1、2、4、8、12、16、24、32 | 相同 | 相同 | 相同 |
| `kept_bytes` | 134,217,728 | 相同 | 相同 | 相同 |
| 4 个 prompt 的输出 | 和冷启动逐字相同 | 相同 | 相同 | |
| 显存 | 21552 MiB | 21554 MiB | 21554 MiB | 21528 MiB |

- decode 吞吐（`bench_serving.py`，1 / 8 / 32 并发）：热改的 168.7 / 1264.9 / 4180.0 token/s，冷启动的 168.7 / 1263.9 / 4160.3。
- 新旧两套 kernel 在这 4 个 prompt 上的输出不同（各在第 41、89、36、66 个字符分开），所以"输出相同"不是因为新代码没跑。

旧树（`f096075`），撤回以后和它自己的冷启动比：身份、启动记录、输出都相同；显存 21486 对 21466 MiB。只有 `kept_bytes` 不同（534,773,760 对 557,842,432）：这是旧树自己的计量问题，FlashInfer 的 workspace 算不算进去取决于初始化顺序，第二批修的正是它，新树两边都是 128 MiB。

### 带在途请求

- **不带参数**：被拒绝，退出码 1。诊断写明前端有 1 个打开的请求、调度器有 1 个在跑的请求和 1 步在途，并说明可以用 `--wait-empty` 或 `--force`。
- **`--wait-empty`**：
  - 等待期间新请求回 503（`the engine is being changed; retry shortly`）；
  - 在途的那个流照常跑完，2500 个 token，以 `[DONE]` 结束；
  - 流结束后变更提交，apply 一共用了 12.5 秒，其中等了 10.5 秒；
  - 提交以后新请求回 200。
- **`--force`**：2.0 秒提交。在途的流在第 531 个 token 收到一个错误帧（503，`ended: the engine is being changed (apply --force)`）后结束。

### 显存

来回 3 轮，每次 apply 以后读一次：旧树 21488 MiB，新树 21554 MiB，三轮都是这两个数。

**热改过的引擎比冷启动的多占 20–26 MiB**（新树 21554 对 21528，旧树 21486 对 21466）。这是个常数，不随 apply 的次数增长。我猜是上一个版本编译好的 kernel 还留在进程里，没有验证。

## 碰到的事

- **身份里有运行目录的路径。** 我第一次对照时冷启动用了另一个 `--run` 目录，身份哈希就不同（`0328755ee417831d`）；换成同一个路径重新冷启动，哈希相同。要比身份，两边的 `--run` 得是同一个路径。
- **静态检查拦下过我两次**，都是模块级的可变对象（`manifest.mutable_global`）：第一批是两个 dict 和一个 list，第二批又是两个 dict。启动或 `check` 时就报，代价很小。
- **一个被接受的配置把引擎压垮了。** `sched.token_budget` 调到 16384 以后，一步 prefill 的瞬时显存超过了状态池之外剩下的显存，worker OOM。预算和显存划分之间没有联动，apply 时不会拦。
- **CI 上 `--force` 的场景偶发失败过一次**（`tests/scenarios/test_m2a.py::test_a_change_applies_to_an_empty_engine_refused_forced_or_waited_for`，`a471a04`）。重跑通过；同一份代码在前一个提交上是绿的；本地单跑 4 次都过；上面试验里的那次 `--force` 也成功。失败原因在 CI 日志里被 pytest 截断了（断言的消息是个 dict），我没拿到，所以没有写进 `doc/framework-bugs/`。
- FlashInfer 的原生依赖同一进程里不能 import 两次。引擎进程没有这个问题（重载只丢单元自己的模块，上面 9 次重载 `kernels.flashinfer` 都正常）；只有 pytest 进程会碰到，靠 `tests/gpu/conftest.py`。

## 为什么没一直用，下次怎么办

- **一张卡放不下。** 引擎按默认的 `--mem-fraction 0.85` 占 21.5 GB。每跑一次 GPU 测试、profile 或 SGLang，都得先停引擎，回来再起就是冷启动。
  - 没试过的办法：把状态池配小（`--mem-fraction`，或直接给页数和槽数），让引擎和测试共存。
- **验证要进到进程里。** kernel 对照参照实现的张量、graph 对 eager 的 logits 和状态、`torch.profiler` 的 trace，经 HTTP 拿不到。这次的开发大半是这类验证，所以大半在 pytest 里。
  - 这一条现在没有办法：跑着的引擎里没有地方做这种检查。
- **我自己的习惯。** 换 RMSNorm 实验代码和换融合 kernel 那两次，引擎刚好停着，我就顺手冷启动了。按上面的试验，起来以后一直留着、改完 apply，是可以的。

下次这样做：

- 工作区用 worktree 或仓库本身，改完先 `apply --dry-run` 看计划，再 apply；
- 压测前后要换代码时，不停引擎，直接 apply；
- 只有要用整张卡的时候才停。

## 没覆盖的

- 改框架本体（应该要求重启引擎）、改权重存储、改拓扑或平台；
- 多 rank、多机；
- SM100 和完整的 64 层模型；
- 引擎里有很多请求时的 `--wait-empty` 和 `--force`（这里只有 1 个）；
- 第一批的改动（新增两个 kernel 单元和它们的依赖库）没有这样试过。
