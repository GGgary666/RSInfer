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
- ~~两端在同一个进程里时，锁测不出来（POSIX 锁按进程算），退回只看心跳。~~ 2026-11-12 换成 OFD 锁以后同一进程的两端也测得出（见下）。

## 审查以后改的（2026-11-12）

Codex 和 Gemini 审了 #55。成立的几条都改了，每条都有在旧代码上失败的测试。项目只支持 Linux，只关于 macOS 的意见不改。

- **撤回构建只杀 Python 父进程**（Codex 复现；Gemini 也提了）。编译器子进程留着，PyTorch 的锁文件（`<build_dir>/lock`）也留着，它等锁时不看持锁的进程还在不在，于是同一份源码以后的每次构建都卡在等锁上，冷启动的加载也一样。现在：
  - 构建进程在自己的会话里。撤回时先给会话里每个进程发 SIGTERM（按 `/proc/<pid>/stat` 里的会话号找），构建进程把它变成 `SystemExit`，PyTorch 的 `finally` 删锁、杀 `ninja`。`change.build_stop_grace_s`（默认 5 秒）以后，会话里剩下的进程全部 SIGKILL；构建进程收尸之前就发，它的会话号这时还不会被复用。
  - 按会话找，不按进程组：`ninja` 把每个编译命令放进它自己的进程组。第一版按进程组发信号，在 4090 D 上撤回以后还有一个 `nvcc` 接着编译。玩具测试里的 "ninja" 现在也把编译器放进自己的进程组，按进程组发的版本在它上面失败。
  - 跑变更的进程结束时，构建进程收到 SIGTERM（`PR_SET_PDEATHSIG`）；进程停机（`ProcessHost.shutdown`）时也先停掉它。
  - `kernels.cuda.ops.build` 在 PyTorch 的锁外面再套一把 `flock`（`only_builder`），持有者怎么结束它都会被放掉。拿到它时如果还有 `lock`，那一定是死掉的构建留下的，删掉。顺带解决了 Gemini 的第 4 条：同机多个 rank 同时构建，在锁外写源码文件。
  - 测试：`tests/framework/change/test_build_steps.py` 里，玩具构建照 PyTorch 的方式拿锁文件、跑一个"编译器"，"编译器"再起一个不理 SIGTERM 的子进程。变更超过准备时限撤回以后，两个编译器进程都不在了，锁文件也没了，同一份源码再 apply 能提交。旧代码上两个编译器进程都还活着。`tests/units/kernels/test_cuda_build.py`：构建进程拿着 PyTorch 的锁被 SIGKILL，下一个构建照样往下走。
- **`itertools.chain` 没有 `close()`**（Gemini）。撤回时 CPython 按引用计数回收，里面的生成器其实也关了，但不该靠这个。换成普通生成器 `_chain`。
- **存活锁用的 POSIX 记录锁属于进程**（Codex 复现；Gemini 也提了）。进程里关掉这个段的任何一个描述符，它在段上的锁就全放掉了，比如被拒的第二个写端关自己的描述符。之后原来的写端一占 GIL，就被当成死了。换成 open file description 锁（`F_OFD_SETLK`/`F_OFD_GETLK`）：锁跟着描述符走，不用 `lseek`，同一进程的两端也互相看得见。代价：不经 exec 的 fork 子进程会共享它（共享内存的描述符在 exec 时关掉）；引擎里没有这样的 fork。测试：`test_stream_liveness.py` 加一种情况，写端的进程先开第二个写端被拒，再占住 GIL，读端不报 `PeerLost`。旧代码上这一条失败。
- **执行时限**（Gemini 的第 5 条，加上用户的决定）。`Execute` 原来用 `supervisor.rpc_timeout_s`（30 秒）。SM100 上 64 层模型醒来要读 55 GB 权重（约 25 秒）、录 12 个 graph 桶（约 6.4 秒），超过 30 秒。随后的 `Abort` 也是 30 秒，参与者 Execute 没返回之前读不到它；执行超过 60 秒，引擎就要求恢复。现在 `change.execute_timeout_s` 默认 3600 秒，执行以后发的 `Abort` 也等这么久；控制面客户端等回复的时限放到 7200 秒（加上 `--wait-empty` 的时间）。测试：`tests/framework/change/test_execute_time_limit.py`，`rpc_timeout_s` 设成 1 秒：执行 3 秒的变更能提交；执行时限 2 秒、执行 3 秒的变更撤回，撤回得到确认，引擎不要求恢复。旧代码上两条都失败。
- **`core.deps` 和 `core.manifest` 不知道 `[unit] build`**（Gemini 的第 1 条）。
  - 影响分析只从入口模块算 import 闭包，构建步骤的模块要是 import 了入口没 import 的源码，改那份源码不会重载这个单元。现在两个模块都算。
  - 静态检查现在也查构建模块存在（`manifest.missing_entry`）、是这个单元自己的（`manifest.cross_unit_import`）。
  - `kernels.cuda` 不受这条影响：它的构建模块就在入口的闭包里。
  - 测试：`tests/units/core/test_deps.py`、`test_manifest.py`，旧代码上 3 条都失败。
  - 不成立的部分：只改 `unit.toml` 里的 `build` 字段本身也会重载，清单一变就算改动。新单元能不能在准备阶段装上（`stageable`）也不用把构建模块算进去，因为构建步骤在自己的进程里、用新代码跑，装新单元之前先跑完。

在 4090 D 上的验证（4 层、dummy 权重、开 graph，`change.prepare_timeout_s` 设成 20 秒）：

| 步骤 | 结果 |
|---|---|
| apply 改过一行注释的 `kernels.cuda` C++ 源码，编译约 55 秒 | 20 秒时准备超时，`aborted`。撤回后 0.5 秒，`nvcc`、`cicc`、`ptxas`、`ninja` 一个也不在了，构建目录里没有 PyTorch 的 `lock`。这期间每 0.5 秒一个请求，67 个全部 200 |
| 引擎外手工构建同一份源码（`python -m kernels.cuda.ops`） | 54.4 秒编完，没有等锁 |
| 再 apply 同一份源码 | `committed`：准备 1.85 秒（构建步骤命中缓存，1.6 秒），暂停 0.02 秒，执行 0.19 秒；输出和 apply 之前逐字相同 |
| `sleep` / `wake` | 962 MiB / 21480 MiB，醒来以后输出相同 |

第一次起这个引擎失败了：GPU 环境的 Python 是 python-build-standalone 的（`uv` 装的），编译时没带 `fcntl.F_OFD_SETLK`，单元加载时报 `AttributeError`。现在按 Linux 的值（36、37）兜底，两种 Python 上 `test_stream_liveness.py` 都过。

### 第二轮（2026-11-12）

Codex 复测了上面两条，都过了；它又找到一条，Gemini 也提了（不挡合并）：

- **跑变更的进程被 SIGKILL 时，构建进程退了，编译器还在**。构建进程收到内核发的 SIGTERM，删锁、杀掉它等的 `ninja`，然后就退出了。`ninja` 起的编译器在自己的进程组里，没人管。负责扫整个会话的 `_stop_build` 在已经死掉的那个进程里。锁放了，旧编译器还在往构建目录里写，下一次构建同一份源码会和它撞上。
  - 现在构建进程发现自己的父进程没了，就自己收拾：先给会话里其余的进程发 SIGTERM，再做清理（删锁），`change.build_stop_grace_s` 以后会话里还剩的全部 SIGKILL。父进程还在时不变，仍由它从外面停。
  - 构建进程原来是一段 `python -c` 的字符串，现在是模块 `core.change.build`（`python -m core.change.build <步骤> <秒数>`）。扫会话的那段代码两边共用。
  - 测试：`test_build_steps.py::test_a_build_step_whose_process_was_killed_takes_its_compilers_with_it`。在真实进程上起变更，等编译器起来，SIGKILL 正在编译的那个进程；它的两个编译器（都不理 SIGTERM）在时限内退出，锁文件没了，其余进程的构建照旧被撤回。改之前第二个编译器一直活着，测试失败。
  - 用真编译器核对（4090 D，没起引擎）：一个替身进程照参与者的方式起 `kernels.cuda.ops:build`，源码改了一行注释，`nvcc` 编到一半时 SIGKILL 替身。

    | | SIGKILL 后 0.2 秒 | 之后 | 锁 |
    |---|---|---|---|
    | 改之前（`d71770b` 的包装） | `sh`、`nvcc`、`gcc`、`cc1plus` 都在 | 8 秒时 `nvcc`、`cudafe++` 还在编 | PyTorch 的锁文件没了，`flock` 放了 |
    | 改之后 | 只剩构建进程自己 | 2 秒内它也退了 | 同上 |

    会话里的进程组也看到了：构建进程和 `ninja` 一组，`sh`、`nvcc`、`gcc`、`cc1plus` 另一组。

Gemini 的另外两条：

- **`--timeout` 超过 3600 时，执行时限没跟着放长**：成立，改了。`deploy/agxinfer.py` 现在把 `change.execute_timeout_s` 设成 `--timeout` 和默认值（3600 秒）里大的那个。`wake` 在执行阶段重读权重，启动要多久它就可能要多久。`test_m2a.py` 有测试。
- **执行失败的那条路没有续租**：不改。执行期间协调者的 `KeepAlive` 排在 socket 里，`Execute` 一返回，同一次轮询接着把它们读出来，每条都续租，然后才查租约。执行短，租约本来就没到期；执行长，排着的 `KeepAlive` 一定有。加一行只是为了对称，没有可观察的差别，框架本体不为这个动。

这一轮没解决的：构建进程自己被 SIGKILL 时（比如 OOM），它起的编译器没人停，编完当前文件自己退出。

还没做（Gemini 和我都提到的）：协调者等一个进程回复时不看它是不是已经死了。上面那条新测试也碰到了：进程被 SIGKILL 以后，变更要等满准备时限才撤回（测试里把时限设成 6 秒）。执行时限长了以后，worker 在执行时崩掉，变更要等满 3600 秒才撤回，这段时间引擎一直暂停。补法是让进程管家的调用在目标进程退出时立刻失败，等用户决定。
