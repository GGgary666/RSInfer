# 准备阶段的构建步骤、共享内存的心跳先问内核（2026-11-11，RTX 4090 D）

讨论里定下的两件框架本体的事，叠在 PR #54 上面：

1. **准备阶段编译**：改了 `kernels.cuda` 的 C++ 源码直接 apply，以前编译发生在暂停期内的重载里，超过执行时限（30 秒），变更被撤回，引擎回了约一分钟的 503。
2. **心跳核实**：进程里一个占住 GIL 超过 `ipc.peer_timeout_s`（2 秒）的调用，会让共享内存流的对端以为这个进程死了（[a-long-gil-hold-looks-like-a-dead-peer](../framework-bugs/2026-11-10-a-long-gil-hold-looks-like-a-dead-peer.md)）。触发它的那一处修过了，机制本身没改。

## 结论

1. **改了 C++ 源码直接 apply 就提交**：编译在准备阶段另起一个进程做（56.8 秒），引擎照常服务。这段时间每 0.5 秒一个请求，134 个里 133 个 200，1 个 503 落在暂停那一下，最慢的一个 0.08 秒。暂停 0.02 秒，执行 0.18 秒，重载从缓存加载只用 0.06 秒。
2. **调度器占住 GIL 5 秒，请求照常完成**。PR #54 的代码上，同样的 5 秒让 worker 以为调度器换了人，请求回 400，说它在那边的东西没了。
3. 框架的测试 310 项、formal 全流程（TLC 的模型和 59 项轨迹回放）、Linux 默认的 pytest、`tests/gpu/` 都通过。

## 改了什么

### 构建步骤（`core.common.manifest`、`core.change.participant`）

- 清单的 `[unit]` 新增可选字段 `build = "模块:函数"`：这个单元有什么慢的东西要先做出来。
- 变更重载这个单元、或者把它加进一个已在服务的进程时，参与者在准备阶段先跑它：
  - 用这次变更的代码（`PYTHONPATH` 指向候选快照）另起一个进程，环境和 worker 一样（同一张卡、同一个 `PATH`），输出写进 worker 的日志；
  - 每个安全点只看一眼它跑完没有，所以它跑多久，旧版本就照常服务多久，也不占这个进程的 GIL；
  - 退出码不是 0，变更失败（`change.build_failed`），这时什么都还没暂停；撤回时把还在跑的进程杀掉；
  - 有新增单元要装的话，构建先跑完再装。
- 进程刚起来时（启动）不跑：那时还没有服务，加载自己会做。
- `UnitSection` 改成不编码默认值：没有构建步骤的清单编出来和以前一样，旧代码也能解开。起因是控制面和进程用的不是同一份框架代码时（控制面跑仓库的代码，进程跑 `--source` 的快照），旧进程不认识新字段，启动失败。
- `kernels.cuda` 声明 `build = "kernels.cuda.ops:build"`；`ops.build` 在没有 CUDA 设备的进程里什么都不做。

### 心跳先问内核（`core.ipc.stream`）

- 共享内存的两端各在环的段上持有一把 POSIX 记录锁（`os.lockf`）：读端锁第 0 字节，写端锁第 1 字节。进程不管怎么结束，内核都会放掉它。
- 心跳超时、锁也没人持有，才算对端不在了。心跳超时但锁还被持有，对端是活着、在忙，接着等。卡死的进程不再由流来判定，交给调用方自己的时限。
- 测锁用 `F_TEST`，只看不拿。拿不到这种锁的平台，或者两端在同一个进程里（POSIX 锁按进程算，自己的锁测不出来），退回只看心跳，和原来一样。
- TCP 不变：本来就由内核判定（keepalive、FIN/RST）。

## 在跑着的引擎上的验证

4090 D，4 层 bf16 随机权重，`--cuda-graph auto --overlap --max-rows 8`。框架本体改了，所以每种代码各冷启动一个引擎。

### 占住 GIL 的调度器

临时检查写在另一份工作区里（`ws_*_probe`），热改上去：调度器的 `step` 里，第 40 步以后、有步在途时，用 `ctypes.PyDLL` 调 libc 的 `usleep` 占住 GIL 5 秒，只做一次。然后发一个生成 96 个 token 的请求。

| 引擎的代码 | 这个请求 | 调度器的日志 |
|---|---|---|
| PR #54（`dev/cli-and-recovery`） | 400，6.24 秒：`results of rank 0 come from a new producer (a new worker process?): what the request had there is gone` | `1 request(s) ended, step(s) [43, 44] given up` |
| 这个 PR | **200，5.59 秒，96 个 token**，文本和前者恢复以后的请求相同 | 只有检查自己的两行 |

两个引擎冷启动后的基线输出相同。撤掉检查以后，输出和基线逐字相同。

### 直接 apply 改过的 C++ 源码

工作区是仓库的副本，`kv_store_cu.py` 的 CUDA 源码里多一行注释（源码摘要变了，就是另一个模块，要重新编译）。一个客户端每 0.5 秒发一个 8 token 的请求，apply：

| | |
|---|---|
| 计划（`--dry-run`） | worker 重载 `core.plan`、`kernels.cuda` |
| 结果 | 提交，命令用了 58.8 秒 |
| 各阶段 | 检查和规划 1.03 秒，**准备 56.81 秒**，排空 0.01 秒，**暂停 0.02 秒**，**执行 0.18 秒**，提交 0.41 秒 |
| worker 的日志 | 构建步骤编译 55.10 秒（`build step kernels.cuda.ops:build done in 56.7s`），暂停期内的重载 `ready in 0.06 s` |
| 期间的请求 | 134 个：133 个 200，1 个 503（暂停那一下），最慢 0.08 秒 |

改回原来的源码再 apply（旧模块在缓存里）：3.5 秒提交，准备 1.64 秒（构建步骤从缓存加载，多数时间是另起的进程 import torch），执行 0.12 秒；身份回到改之前的值，输出和基线相同。

以前同样的 apply：执行时限 30 秒到了，变更撤回，引擎回了 54 秒的 503（[real-kernels-kv-store-indices.md](real-kernels-kv-store-indices.md)）。

## 测试

- `tests/framework/change/test_build_steps.py`（进 CI，两台假主机上的真实进程）：
  - 构建步骤每个进程跑一次，用的是新身份的快照；跑的时候旧版本照常服务、没有暂停，准备阶段至少等了它 1.5 秒；
  - 失败的构建步骤让变更在暂停之前失败，诊断说哪个单元、退出码 3，各进程还是旧身份、照常服务；
  - 没有构建步骤的单元不起进程；
  - 清单里写错格式的 `build` 被拒。
- `tests/framework/ipc/test_stream_liveness.py`（进 CI）：子进程占住 GIL 4 倍的 `peer_timeout_s`。
  - 读端占着：写端写满了环就等，放开后 400 条记录按序到达；
  - 写端占着：读端等着，不报 `PeerLost`；
  - 占着 GIL 的读端被 SIGKILL：写端在 `peer_timeout_s` 加 2 秒以内报 `PeerLost`。
  - 前两条在旧代码上都报 `PeerLost` 失败。
- `uv run pytest -n 8 tests/framework`：310 passed；`uv run pyright`：0 错误。
- formal 全流程：`uv run python tests/framework/formal/run.py --bootstrap --replay --java <JRE 17>`，全部模型检查通过，轨迹回放 59 passed。这台机器原来没有 Java，临时下载了 Temurin JRE 17.0.21 到临时目录。

## 没做的、要注意的

- 构建步骤只在"重载"和"加进已在服务的进程"时跑。启动时不跑，第一次启动照旧在加载里编译，那时也没有服务要保护。
- 构建步骤每次都另起一个 Python 进程，光 import torch 就一两秒。只有改了声明它的单元的代码才会跑。
- 醒来（`holder.sleep` 改回 false）时读权重仍在暂停期里（PR #54），没有改成构建步骤：读进来的权重要放在这个进程里，另起的进程帮不上。要挪出暂停期，得让单元在准备阶段、在自己的进程里分片读，是另一件事。
- 心跳线程本身没挪走。卡死而活着的对端，流不再替调用方判定。
- 两端在同一个进程里时，锁测不出来（POSIX 锁按进程算），退回只看心跳。引擎里流的两端总在不同的进程里。
