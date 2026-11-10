---
name: agxinfer
description: Working in the agxinfer repository - where each component, unit, test and document lives (the directory map), and how to develop on a running engine by hot-applying changes (check, apply --dry-run, apply, --wait-empty, --force), what a normal refusal looks like, and how to report a bug of the update mechanism in doc/framework-bugs/. Use before changing code, tests or docs in this repository, or when starting, changing or debugging a running engine.
---

# agxinfer：目录地图与热改开发

规矩在 [AGENTS.md](../../../AGENTS.md)；这里是东西在哪，以及怎么在跑着的引擎上改东西。本文件的目录地图要和代码保持一致（AGENTS.md「文档是开发笔记」）。

## 目录地图

### 顶层

| 路径 | 内容 |
|---|---|
| `engine/` | 引擎源码树（框架本体 + 将来的单元） |
| `tests/` | 全部测试 |
| `doc/` | 设计文档、规格、评估记录 |
| `.github/workflows/ci.yml`、`.github/scripts/framework_changed.py` | CI：`uv sync --python 3.12 --extra cpu`、`uv run pyright`、`uv run pytest -n auto`（Ubuntu，pytest-xdist 按核数并行）。PR 每次 push 只跑一遍；push 只在 `master` 上触发。PR 和 push 没碰框架本体（`engine/framework.toml` 列出的模块）、`tests/framework/`、共用的测试设置、依赖和 `.github/` 时，跳过 `tests/framework/` |
| `.github/workflows/notify.yml`、`.github/scripts/notify.py` | 每个合进 `master` 的 PR 都由 LLM 写一句话摘要发到通知群；密钥 `PR_NOTIFY_WEBHOOK`、`PR_NOTIFY_SECRET`、`LLM_API_KEY`、`LLM_BASE_URL`，可选变量 `LLM_MODEL` |
| `pyproject.toml`、`uv.lock` | 依赖与 pytest / pyright 配置 |
| `.claude/skills/agxinfer/SKILL.md` | 本文件：目录地图与热改开发。Claude Code 按 skill 自动加载，其他 agent 照 AGENTS.md 的指引读它。用户同意为它新开 `.claude/` |
| `deploy/` | 部署工具，不属于 pytest，用户同意为它新开这个顶层目录。`qwen38.py`：在一台机器上起 Qwen3.8（`plan` / `serve` / `chat` / `checkpoint`），说明见 `deploy/README.md`；在 pyright 的范围内 |

### `engine/`

不安装成包（`[tool.uv] package = false`），靠 `pythonpath = ["engine", "."]` 导入。导入名是 `core.*`、`unit.*`、`platforms.*`，**不是** `engine.*`。运行时进程从按身份物化的快照目录加载这棵树（`doc/specs/nodeagent.md` §3）。

| 路径 | 内容 | 规格 |
|---|---|---|
| `engine/framework.toml` | 哪些模块属于框架本体；改它们只能重启引擎，不能热重载 | `doc/specs/README.md` |
| `engine/core/common/` | 标识、单元清单解析、消息信封、`Ok`/`Failed` 结果、可调参数 | `doc/specs/README.md` |
| `engine/core/context/` | 配置项、运行时 flag、资源、capture 依赖记录 | `context.md` |
| `engine/core/ipc/` | RPC、pubsub、step stream（同机共享内存环 / 跨机 TCP）、open records、宿主替单元持有的数据通道 `proc.streams@1`（`channels.py`） | `ipc.md` |
| `engine/core/nodeagent/` | 每台机器一个的节点代理（`python -m core.nodeagent`） | `nodeagent.md` |
| `engine/core/supervisor/` | 进程管家、拓扑、管家与节点代理之间的消息 | `supervisor.md` |
| `engine/core/change/` | 变更事务：协调者、参与者、事件日志、blob 存储、队列 | `change.md` |
| `engine/core/control/` | 控制面进程（`python -m core.control serve`）、控制接口的消息和客户端（`python -m core.control.client`） | `control-plane.md` |
| `engine/unit/runtime/` | 单元运行时：加载 / 重载 / 撤销、`ProcessHost`、注册表、静态 import 图 | `unit-runtime.md` |
| `engine/platforms/` | 平台接口 `base.py`；CI 用的 `fake/`；真张量、只做 eager 的 `torch_eager/`（CPU 或进程自己的 CUDA 卡）；`cuda_graph/` 复用 torch 分配并提供进程 capture stream、warm-up、capture/replay/close。合法平台名 `PLATFORMS`（`__init__.py`），由拓扑角色的 `platform` 决定 | `platform.md`、`cuda-graph.md` |

控制面单元（单元，不属于框架本体，不在 `framework.toml` 里）：

| 路径 | 内容 | 规格 |
|---|---|---|
| `engine/core/identity/` | 系统身份 `core.identity` | `control-plane.md` §2 |
| `engine/core/manifest/` | 清单与静态检查 `core.manifest` | `control-plane.md` §3 |
| `engine/core/deps/` | 影响分析（动作计划、dry-run）`core.deps` | `control-plane.md` §4 |
| `engine/core/api/` | 控制接口 `core.api` | `control-plane.md` §5 |
| `engine/core/sourcetree.py` | 读源码树（清单、模块）的库，`core.manifest` 和 `core.deps` 共用；不是单元 | |
| `engine/core/contracts/` | 单元目录之外的共享接口库，避免 `manifest.cross_unit_import`；只放实际交接的数据和必要调用，不 import torch，不放内部算法和整对象镜像；修改原则见 AGENTS.md「约定」 | `stage2-contract.md` §3 |

调度宿主里的调度单元（阶段 2 第 3 步，来源、行为和已知缺口见 `engine/core/sched/README.md`）：

| 路径 | 内容 |
|---|---|
| `engine/core/sched/` | 调度器 `core.sched`（`proc.loop@1`）：两个决策点、在途步队列（深度 1，开 `sched.overlap` 为 2）、多 rank、出错不冻结（`loop.py`）；变更前排空（`drain.py`，`doc/specs/empty-apply.md`） |
| `engine/policies/` | 决策点的策略单元：`fcfs`（`policy.order@1`）、`prefill_first`（`policy.compose@1`），接口在 `core/contracts/policy.py` |
| `engine/core/state/alloc/` | 联合账本 `core.state.alloc`（`ledger@1`，接口在 `core/contracts/ledger.py`）；`prefix.py` 是账本内部的联合前缀匹配算法 |

阶段 2 的文本前端：`engine/frontend/http/`、`tokenizer/`、`detok/` 是前端的三个单元（`frontend.http` 也是变更时引擎的门：变更期间新请求回 503，`api.py` 的 `Gate`），`frontend/interfaces.py`（它们之间的本地接口）和 `transport.py`（前端的两条记录流）是它们共用的库；`engine/sched/reqs/` 是调度宿主里的请求表状态单元 `sched.reqs`。来源、行为和运行方法见 `engine/frontend/README.md`。

worker 里的执行层单元（阶段 2，eager 和 TP1 decode graph，`stage2-contract.md` §6）：

| 路径 | 内容 |
|---|---|
| `engine/core/kernels/` | kernel 目录 `core.kernels`（`kernels.catalog@1`）：登记 definition 和 solution，按 shape 和平台能力选，没有合适的就用参照实现 |
| `engine/kernels/reference/` | `kernels.reference`：Qwen3.8 要的全部 definition 及 PyTorch 参照（`ops.py`），从 SGLang 搬运，签名和状态排布照 SGLang |
| `engine/core/state/weights/` | `core.state.weights`（`state.weights@1`）：参数存储、safetensors 读取、dummy 权重、`refit` |
| `engine/models/qwen38/` | `models.qwen38`（`model@1`）：SGLang `Qwen3_5ForCausalLM` 的搬运，参数名、`load_weights`、前向 |
| `engine/layers/` | `layers.linear`、`embedding`、`norm`、`rope`、`activation`、`sampler`；`layers.attn`、`layers.gdn` 是 attention / GDN 外壳加 backend（SGLang 的 `TorchNativeAttnBackend`、`GDNAttnBackend`）；`solutions.py` 是这两个 backend 共用的库：按 `attn.*` / `gdn.*` 的 backend 配置在启动时固定每个阶段的 kernel |
| `engine/state/` | 存储单元 `state.kv_paged`（`MHATokenToKVPool`）、`state.gdn_slots`（`MambaPool`） |
| `engine/core/state/holder/` | `core.state.holder`（`state.holder@1`）：按 SGLang 的分法定 KV 页数和 GDN 槽数（`mem_fraction_static`、`mamba_full_memory_ratio`、`max_mamba_cache_size`），启动时定，阶段 2 不在线扩容 |
| `engine/worker/batch/` | 持久 batch `worker.batch`（`HybridReqToTokenPool` 的表） |
| `engine/executors/pytorch/` | `executors.pytorch`（`executor@1`）：worker 唯一的 executor，SGLang `ModelRunner` 的流程，在 `torch_eager` 和 `cuda_graph` 上都用它；required 缺图在状态写入前拒绝 |
| `engine/core/plan/` | `core.plan`（`plan@1`）：eager（bucket 0）和 TP1 decode graph；`graph.py` 持有固定输入、依赖守卫及候选构建验证。启动/暂停期录制，支持 auto/off/required、bucket、显存预算，日志为 `core.plan: {json}` |
| `engine/core/worker/` | worker 循环 `core.worker`（`proc.loop@1`）：读计划、交给 `executor@1`、按计划顺序回 `Sampled` / `Resync` / `StepFailure`，启动时和结果换了读端时报 `Capacity` |

从 SGLang（`e838360`）搬运的代码在文件头注明出处，各目录带一份 `LICENSE.sglang`。改这些单元时先看 SGLang 的对应实现，能搬运的就搬运后适配，不自己另写。

每个单元目录下有一份 `unit.toml`。将来的单元（调度策略、KV cache、模型等）也放在 `engine/` 下，同样不列进 `framework.toml`。

### `tests/`

- `tests/__init__.py`：让 `tests` 成为包。所有测试辅助代码都用 `tests.…` 绝对导入。
- `tests/conftest.py`：autouse 的 `_isolate_imports`，对所有测试生效：每个测试后恢复 `sys.path`、`sys.modules` 和各个包的 `__path__`（toy unit 被导入进全进程共享的 `sys.modules`；进程内的控制面会让包先去快照里找模块）。
- `tests/framework/`：框架本体（`engine/framework.toml` 列出的模块）的测试。
  - `conftest.py`：`pkg`、`snapshot` fixture。
  - `support/`：通用工具 `spec`、`manifest_toml`、`write_tree`、`PROC`、`events`、`GENERIC_UNIT`。
  - `unit_runtime/`、`context/`、`ipc/`、`platforms/`、`change/`、`cluster/`：按组件划分。`ipc/test_location_transparency.py` 扫描整个 `engine/`，禁止写死本机地址或用 PID 当地址（P7）。
  - `adversarial/`：对抗与故障边界测试，对应 `doc/evaluations/` 的记录。
  - `formal/`：TLA+ 有限模型（`models/`）、runner（`run.py`）、对真实实现的轨迹回放和契约测试。细节见 `tests/framework/formal/README.md`。
- `tests/units/`：单元的测试（不是框架本体的测试）。`units/core/`：控制面单元 `core.identity`、`core.manifest`、`core.deps`，在小源码树上逐条测规则，不起进程。`units/contracts/`：接口库 `core.contracts`（编码往返、单元能 import 它、每个模块只 import 到本接缝的模块）。
- `tests/units/scheduler/`：调度器、账本和前缀匹配算法的测试。`support.py` 把调度宿主的单元放进进程内的单元运行时，配上线程上的夹具 worker 和充当前端的客户端；`test_processes.py` 用真实控制面起调度宿主和两个夹具 worker 进程。`units/schedfix/` 是测试夹具单元（夹具 worker），真实进程测试把它拷进工作区。
- `tests/units/frontend/`：前端和请求表的测试。`support.py` 提供字节 tokenizer、脚本调度器（真请求表、固定回复，测试夹具）和只回答 Lookup 的假进程管家；`demo.py` 起一个无模型的 HTTP 服务。
- `tests/units/kernels/`：kernel 目录的选择和重载，参照 kernel 对照 HF `qwen3_5`（要 `--extra cpu`）。
- `tests/units/worker/`：worker 循环。先对着假 executor 测循环本身，再把 worker 的全部单元加上循环放到线程上（`support.py` 的 `RealWorker`），接在 `tests/units/scheduler/support.py` 的真调度器下面，贪心生成和 HF 逐 token 相同。
- `tests/units/models/`、`state/`、`executors/`：模型和权重存储、存储单元和持久 batch、整个 worker 栈对照 HF（逐 token）和几组对比。`tests/units/support/worker.py` 提供：worker 全部单元装进一个 `UnitRuntime`（`Worker`）、小尺寸 HF 参照模型，以及按调度器的方式写每步记录的测试夹具 `Driver`。
- `tests/units/support/tokenizer.py`：测试用的真 HF tokenizer（词级、带 Qwen 的 chat 标记和模板）。
- `tests/ci/`：CI 自己的工具（`.github/scripts/`）的测试，比如判断一个 PR 有没有碰到框架本体。它既不是框架本体，也不是单元或整个引擎，所以单独一个目录。
- `tests/gpu/`：要 GPU、手动跑（由 `addopts` 排除）。`test_qwen38_partial.py`：Qwen3.8-27B 真实结构的前 4 层，worker 栈对照 HF；`test_qwen38_serve.py`：用 `deploy/qwen38.py serve` 起整个引擎，经 HTTP 对照 HF（`QWEN38_MODEL` 指向完整模型就是 SM100 上的端到端验证）；`test_qwen38_overlap.py`：worker 栈重叠一步（下一步在上一步跑完之前提交）和逐步提交采到的 token 相同，步与步之间设备空得更少；`support.py`：生成部分层数的 checkpoint、HF 模型。24 GB 的卡就够。
  - `test_cuda_graph.py`：capture/replay、状态/padding、HF、refit、依赖失效、候选失败清理、overlap 和多次重录显存；`test_cuda_graph_apply.py`：真实控制面 kernel apply、失败配置撤回、恢复及冷启动对照；配置变更用例核对 plan 代数和实际 capture 次数，区分重录与复用。`test_qwen38_partial.py` 同时覆盖 graph/eager；`tests/units/executors/test_cuda_graph.py` 是无需 GPU 的配置/守卫测试。
  - `bench_cuda_graph.py`：手动 decode 吞吐 benchmark，对比部署 eager、同 kernel eager 和 graph，输出每次测量与环境 JSON；不作为 pytest 的性能阈值断言。
  - `test_cuda_graph_adversarial.py`：A/B/C 共用的 GPU 验收，检查数值、实际 CUDA replay，以及 catalog 变更后在写状态之前拒绝旧图；实验分支、结果与复现命令见 `doc/evaluations/cuda-graph-abc.md`。
- `tests/scenarios/`：整个引擎（框架本体 + 单元）的测试：modules.html 的 9 个场景、命令行 `serve`、覆盖检查和拆除检查（roadmap 阶段 1 的验收）。
  - `test_m2a.py`：阶段 2 的 M2a，在 CPU 上用 `deploy/qwen38.py serve` 起整个引擎（前端、调度宿主、worker，小随机模型、真 HF tokenizer），经 HTTP 的文本和 HF 逐字相同；`deploy.py` 是它和 `tests/gpu/` 共用的起停辅助。
  - `units/scn/`：场景测试用的单元（测试夹具，写法和真单元一样，但只做场景需要的那一步）；`harness.py` 把它们和本仓库的 `engine/` 一起拷进工作区，但不拷 `engine/` 里控制面以外的单元（前端、请求表、kernel、模型等）。场景拓扑的角色不列单元，那些单元会在每个场景进程里加载；kernel 单元每次启动还要 import torch。
- 测试辅助模块（不以 `test_` 开头，pytest 不收集）：

  | 模块 | 提供 |
  |---|---|
  | `framework/cluster/harness.py` | 测试事件循环里的真实控制面 + 子进程里的节点代理（`Cluster`，`async with cluster(...)`） |
  | `framework/change/engine.py` | 基于 cluster harness 的协调者测试台（`Engine`，`async with engine(...)`） |
  | `framework/change/harness.py` | 进程内 Coordinator + 两个真实 `ProcessHost`，可在传输边界丢请求 / 回执（`Harness`、`Capture`、`OLD`、`harness`） |
  | `framework/adversarial/peer.py` | 单个真实 `ProcessHost`，逐条投递参与者消息（`Peer`、`peer`） |
  | `framework/ipc/peers.py` | IPC 测试在子进程中运行的代码 |
  | `scenarios/harness.py` | 进程内的控制面（带控制面单元）+ 两台假主机，通过控制接口改工作区、`apply`（`Scenario`、`async with scenario(...)`、`build_workspace`、`rungs`） |
  | `scenarios/coverage.py` | 覆盖检查要数的东西：6 种边、动作阶梯 0–6、3 种模块 |

- 被多个测试文件共用的代码放进上面这类辅助模块（同目录或 `support/`），**不要从另一个 `test_*.py` 里导入**。fixture 以 `from … import harness as harness` 的形式在测试文件中重新导出。

### `doc/`

- `doc/modules.html`：**设计真源**。冲突时：`modules.html` > `v0-plan.md` > `discussion-summary.md`。
- `doc/roadmap.md`：阶段和验收。
- `doc/specs/`：框架本体七个组件的规格，以及控制面（`control-plane.md`：控制面进程和四个控制面单元）、变更只作用于空引擎（`empty-apply.md`：除了只改 live 键，每次变更前排空整个引擎，有请求时默认拒绝；取代 `state-changes.md` 的排空与版本迁移）、阶段 2 的单元契约（`stage2-contract.md`：单元清单、接口、分工、里程碑、分支和合并；草案）、CUDA Graph 实现方案（`cuda-graph.md`：TP1 decode graph、最小改动与 fail fast、暂停期重录、首版实现选择和验收矩阵）；`README.md` 是总览和公共约定（标识、清单、信封、结果、测试约定）。
- `doc/v0-plan.md`、`doc/discussion-summary.md`：v0 计划、讨论纪要。
- `doc/evaluations/`：历次修复 / 对抗测试的评估记录，属于历史，不随代码更新。
  - `doc/evaluations/gpu-probes/`：阶段 1′ 的 GPU 探针脚本（`run.sh` 一次跑完）和原始输出 `logs/`，结论在 `doc/evaluations/gpu-probes.md`。探针用完就扔，不进骨架；不在 pyright 和 pytest 的范围内（要 GPU、torch 和第三方 kernel 库）。
- `doc/framework-bugs/`：热改时发现的更新机制 bug 的报告，格式见其中的 `README.md`。
- `doc/archive/`：旧版文档，不再维护。
- `doc/figures/`：图片。

## 在跑着的引擎上热改

AGENTS.md 规定默认这样开发：把引擎拉起来，改完用 `apply` 生效。这样开发本身就是对热更新机制（变更协议、单元运行时的重载、`core.deps` 的计划）的测试。碰到机制本身的问题，按 [`doc/framework-bugs/README.md`](../../../doc/framework-bugs/README.md) 写报告；实在绕不过去，就重启引擎，退回每次冷启动的开发方式。热改不代替提交前的 pyright、pytest。

### 起一个引擎

按这次要改、要看的东西起，需要什么就起什么。下面只是几个例子，没有推荐的哪一种：

- 用 `deploy/qwen38.py`（说明在 `deploy/README.md`）在一台机器上起 Qwen3.8 的整个引擎：前端、调度宿主、一个 worker。例如：
  - 真权重：`uv run python deploy/qwen38.py serve --model <模型目录>`；
  - 只要前几层、dummy 权重：再加 `--layers N --dummy`；
  - 整个引擎跑在 CPU 上：加 `--device cpu`，模型可以用 `deploy/qwen38.py checkpoint` 生成的小随机模型（`tests/scenarios/test_m2a.py` 就是这样起的）。
- 直接用控制面，起任意拓扑和单元组合（单机时 `--spawn-agents` 在本机给拓扑里每台主机起节点代理）：

  ```sh
  PYTHONPATH=engine uv run python -m core.control serve --source <工作区> --topology <topology.json> \
      --state-dir <目录> --host <HostId> --bind <地址> [--config <config.json>] --spawn-agents
  ```

- 测试里的起法也可以参考：`tests/scenarios/harness.py`（进程内控制面加两台假主机）、`tests/units/scheduler/test_processes.py`（真实控制面加夹具 worker 进程）。

起来以后：
- 控制面地址：`deploy/qwen38.py` 打印的 `engine up: ... (control <地址>)`，也写在 `<run>/control/control.json` 的 `control` 里；
- 引擎的源码是 `--source` 指向的工作区里的 `engine/`（`deploy/qwen38.py` 默认是仓库本身），各进程从按身份物化的快照目录加载它；
- 各进程的日志：`<快照根目录>/.run/<角色>-<编号>-<n>/log`（`deploy/qwen38.py` 起的在 `<run>/host-a/snapshots/.run/` 下）；`serve` 自己的输出里有控制面和节点代理的日志。

### 改一次

在工作区里改文件，然后（`<地址>` 是控制面地址）：

```sh
export PYTHONPATH=engine
C="uv run python -m core.control.client --control <地址>"
$C check --source <工作区>              # 静态检查：清单、服务、跨单元 import、配置键；不排队、不碰引擎
$C apply --source <工作区> --dry-run    # 影响分析：哪些进程重载哪些单元、要不要重启引擎、notes
$C apply --source <工作区> --intent "为什么改"
$C apply --config '{"sched.overlap": true}'   # 只改配置，不改代码
$C status                               # 也可以是 introspect（单元、配置、谁读了哪个键）、logs（变更日志）
```

- `apply` 默认打印摘要（状态、身份、notes、每个进程动了什么、诊断），`--json` 打印完整结果；退出码 0 表示提交了（`--dry-run` 时是规划成功）。
- 变更只作用于空引擎（`doc/specs/empty-apply.md`）：除了只改 live 键，引擎里有请求时 `apply` 被拒，诊断里写着有多少请求。接着：
  - `--wait-empty [秒]`：前端关门（新请求回 503），等里面的跑完再 apply，默认最多 3600 秒，到了就失败、什么都不中止；
  - `--force`：立刻结束里面的请求（客户端收到 503），再 apply。
- 改完用请求验证：`deploy/qwen38.py chat ...`，或直接对前端发 HTTP；需要时对照冷启动的结果。

### 哪些是正常的，哪些可能是 bug

正常，按提示做就行：
- 引擎里有请求时被拒；worker 出错以后只有 `--force` 能清空；
- 改了框架本体（`engine/framework.toml` 列的模块）或控制面单元的原生扩展：`escalate.restart_engine`，只能重启引擎（AGENTS.md「框架改动」）；
- 改了原生扩展、拓扑、放置或平台：变更自动重启对应的进程；
- 改了权重存储 `core.state.weights` 的代码：在暂停期内从 checkpoint 重新读，大模型要几分钟；`refit` 换过的权重回到 checkpoint；
- 静态检查不过（`check` 或 `apply` 的诊断写明了哪条规则）。

可能是更新机制的 bug，写报告：
- 提交了，可跑的还是旧代码，或者新旧代码混着跑；
- 提交以后请求出错、引擎停止服务，或者输出和用同一份代码冷启动的不一样；
- 计划不对：`--dry-run` 漏了该重载的单元、重载了不相干的、该要求重启的没要求（或反过来）；
- 撤回（`aborted` / `failed`）以后行为和变更之前不一样，或者引擎进入要求恢复的状态；
- 同一个改动 apply 两次结果不同，或者 `--dry-run` 的计划和实际执行的不一样；
- 进程、显存、文件句柄这类资源在多次 apply 之后一直涨。

### 踩过的坑

- 停引擎要按 `serve` 进程的精确 PID 发 SIGTERM（`deploy/qwen38.py` 起的子进程在命令行退出时会跟着收到 SIGTERM）；别用 `pkill -f` 之类按命令行匹配，它会匹配到发命令的 shell 自己。
- GPU 上的第一步可能很慢；超时都按 `--timeout` 放长了。
- `deploy/qwen38.py checkpoint` 生成的 checkpoint 很大（27B 结构的 4 层约 7.6 GiB），用完删掉；快照目录是只读的，删之前先 `chmod -R u+w`。
