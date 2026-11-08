# AGENTS.md

给在本仓库工作的 agent：东西在哪、怎么验证、改目录结构的规矩。项目是什么见 [README.md](README.md)。

## 目录结构规则（最重要）

- **如无必要，尽量不要新增、删除、重命名或移动顶层目录**（`engine/`、`tests/`、`doc/`、`.github/`），除非用户明确要求。
- **新增或移动任何目录之前，先判断它是不是某个已有目录的子集。** 是子集就放进那个目录，不要和它并列。
  - 例：formal 测试首先是测试，而且测的是框架本体，所以在 `tests/framework/formal/`，不是顶层的 `formal-tests/`。
  - 例：将来给单元写的测试是测试但不是框架本体的测试，所以放在 `tests/` 下、与 `tests/framework/` 并列（如 `tests/units/`），不要混进 `tests/framework/`。
- 优先使用已有目录；只有确认已有目录都不合适时才新建，并在说明里写清楚理由。
- 移动、重命名、拆分、合并目录前，先把方案告诉用户并得到确认。改完要同步：
  - 用 `git mv`，保留历史；
  - 绝对导入 `tests.framework.…`，包括写在字符串里的代码（toy unit 源码、子进程的 `python -c "..."`）；
  - 用 `Path(__file__).parents[n]` 算出的路径：`tests/framework/cluster/harness.py` 的 `REPO`、`ipc/test_shm_ring_wrap.py`、`ipc/test_location_transparency.py`、`formal/run.py` 和 `formal/test_*.py`；
  - `pyproject.toml`（`testpaths`、`addopts`、pyright `include`）和 `.github/workflows/ci.yml`；
  - 本文件的目录地图。其他文档不必逐个追改（见下一节）。

## 文档是开发笔记

- `doc/` 下的文档和各目录的 README 是带日期的开发笔记，记录的是写下当时的想法，不是必须一直保持正确的规范。
- 改代码时**不要**连带更新所有引用到它的文档，也不要为了让旧文档保持最新花大量精力。需要记录新的结论时，写一份带日期的新文档；新文档可以取代旧文档。
- 只有两类文档需要和代码保持一致：这次改动直接对应的文档（例如你正在写的设计或评估记录），以及本文件。

## 目录地图

### 顶层

| 路径 | 内容 |
|---|---|
| `engine/` | 引擎源码树（框架本体 + 将来的单元） |
| `tests/` | 全部测试 |
| `doc/` | 设计文档、规格、评估记录 |
| `.github/workflows/ci.yml` | CI：`uv sync --python 3.12`、`uv run pyright`、`uv run pytest -n auto`（Ubuntu，pytest-xdist 按核数并行） |
| `.github/workflows/notify.yml`、`.github/scripts/notify.py` | 每个合进 `master` 的 PR 都由 LLM 写一句话摘要发到通知群；密钥 `PR_NOTIFY_WEBHOOK`、`PR_NOTIFY_SECRET`、`LLM_API_KEY`、`LLM_BASE_URL`，可选变量 `LLM_MODEL` |
| `pyproject.toml`、`uv.lock` | 依赖与 pytest / pyright 配置 |

### `engine/`

不安装成包（`[tool.uv] package = false`），靠 `pythonpath = ["engine", "."]` 导入。导入名是 `core.*`、`unit.*`、`platforms.*`，**不是** `engine.*`。运行时进程从按身份物化的快照目录加载这棵树（`doc/specs/nodeagent.md` §3）。

| 路径 | 内容 | 规格 |
|---|---|---|
| `engine/framework.toml` | 哪些模块属于框架本体；改它们只能重启引擎，不能热重载 | `doc/specs/README.md` |
| `engine/core/common/` | 标识、单元清单解析、消息信封、`Ok`/`Failed` 结果、可调参数 | `doc/specs/README.md` |
| `engine/core/context/` | 配置项、运行时 flag、资源、capture 依赖记录 | `context.md` |
| `engine/core/ipc/` | RPC、pubsub、step stream（同机共享内存环 / 跨机 TCP）、open records | `ipc.md` |
| `engine/core/nodeagent/` | 每台机器一个的节点代理（`python -m core.nodeagent`） | `nodeagent.md` |
| `engine/core/supervisor/` | 进程管家、拓扑、管家与节点代理之间的消息 | `supervisor.md` |
| `engine/core/change/` | 变更事务：协调者、参与者、事件日志、blob 存储、队列 | `change.md` |
| `engine/core/control/` | 控制面进程（`python -m core.control serve`）、控制接口的消息和客户端（`python -m core.control.client`） | `control-plane.md` |
| `engine/unit/runtime/` | 单元运行时：加载 / 重载 / 撤销、`ProcessHost`、注册表、静态 import 图 | `unit-runtime.md` |
| `engine/platforms/` | 平台接口 `base.py`；CI 用的 `fake/` | `platform.md` |

控制面单元（单元，不属于框架本体，不在 `framework.toml` 里）：

| 路径 | 内容 | 规格 |
|---|---|---|
| `engine/core/identity/` | 系统身份 `core.identity` | `control-plane.md` §2 |
| `engine/core/manifest/` | 清单与静态检查 `core.manifest` | `control-plane.md` §3 |
| `engine/core/deps/` | 影响分析（动作计划、dry-run）`core.deps` | `control-plane.md` §4 |
| `engine/core/api/` | 控制接口 `core.api` | `control-plane.md` §5 |
| `engine/core/sourcetree.py` | 读源码树（清单、模块）的库，`core.manifest` 和 `core.deps` 共用；不是单元 | |
| `engine/core/contracts/` | 阶段 2 单元之间的接口（每步记录、状态、kernel 目录、plan 和 backend、权重和模型）；库，不是单元也不属于框架本体，接口两边的单元都 import 它；不 import torch。改接口要接口两边的负责人都审过 | `stage2-contract.md` |

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
- `tests/units/`：单元的测试（不是框架本体的测试）。`units/core/`：控制面单元 `core.identity`、`core.manifest`、`core.deps`，在小源码树上逐条测规则，不起进程。`units/contracts/`：接口库 `core.contracts`（编码往返、联合前缀匹配、单元能 import 它）。
- `tests/scenarios/`：整个引擎（框架本体 + 单元）的测试：modules.html 的 9 个场景、命令行 `serve`、覆盖检查和拆除检查（roadmap 阶段 1 的验收）。
  - `units/scn/`：场景测试用的单元（测试夹具，写法和真单元一样，但只做场景需要的那一步）；`harness.py` 把它们和本仓库的 `engine/` 一起拷进工作区。
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
- `doc/specs/`：框架本体七个组件的规格，以及控制面（`control-plane.md`：控制面进程和四个控制面单元）、状态单元的变更（`state-changes.md`：排空与版本迁移，只有规格）、阶段 2 的单元契约（`stage2-contract.md`：单元清单、接口、分工、里程碑、分支和合并；草案）；`README.md` 是总览和公共约定（标识、清单、信封、结果、测试约定）。
- `doc/v0-plan.md`、`doc/discussion-summary.md`：v0 计划、讨论纪要。
- `doc/evaluations/`：历次修复 / 对抗测试的评估记录，属于历史，不随代码更新。
  - `doc/evaluations/gpu-probes/`：阶段 1′ 的 GPU 探针脚本（`run.sh` 一次跑完）和原始输出 `logs/`，结论在 `doc/evaluations/gpu-probes.md`。探针用完就扔，不进骨架；不在 pyright 和 pytest 的范围内（要 GPU、torch 和第三方 kernel 库）。
- `doc/archive/`：旧版文档，不再维护。
- `doc/figures/`：图片。

## 常用命令

```sh
uv sync --python 3.12
uv run pyright                       # strict；检查 engine/ 和 tests/（含 formal）
uv run pytest                        # 默认测试；不收集 tests/framework/formal；串行
uv run pytest -n 8                   # 全量并行（pytest-xdist）：约 30 秒，串行约 165 秒
uv run pytest tests/framework/ipc -q # 只跑某个目录
# 起一个引擎（单机：--spawn-agents 在本机给拓扑里每台主机起节点代理），再用客户端操作它
PYTHONPATH=engine uv run python -m core.control serve --source <工作区> --topology <topology.json> \
    --state-dir <目录> --host <HostId> --bind <地址> [--config <config.json>] --spawn-agents
PYTHONPATH=engine uv run python -m core.control.client --control <地址> status|check|apply|introspect|logs
# formal 全流程：需要 Java 11+；首次 --bootstrap 下载并校验 TLC 1.7.4
uv run python tests/framework/formal/run.py --bootstrap --replay [--java /path/to/java]
```

- 每次提交前跑 `uv run pyright` 和 `uv run pytest`（全量用 `-n 8` 并行；只跑一两个文件时别加 `-n`，起 worker 要 5 秒左右）。改了 `core.change`、`core.supervisor`、`core.nodeagent`、`unit.runtime.host` 或 `tests/framework/formal/` 时，也跑 formal 全流程。
- `uv run pytest` 跳过 formal 靠的是 `pyproject.toml` 里 `addopts` 的 `--ignore=tests/framework/formal`；显式给出路径（`uv run pytest tests/framework/formal`）不受影响。
- `pyproject.toml` 的 `timeout = 120` 是 pytest-timeout 给**单个测试**的上限，不是整个套件的。不要给 pytest 或 CI job 加总时长上限。
- 生成物不进仓库：`tests/framework/formal/.tools/`、`.runs/` 已被忽略。

已知与环境有关的失败：`ipc/test_streams.py::test_slow_consumer_blocks_the_producer[tcp]` 在 macOS 上偶发失败，Linux 和 CI 上正常。

## 约定

- 文档用中文；代码、注释、docstring 用英文。
- 测试不用 `xfail`，也不按平台或缺少工具自动 `skip`。
- 测试用 `@pytest.mark.covers(edges=..., ladder=..., kinds=...)` 声明自己走到了哪些边、动作阶梯的哪几级、哪些模块种类，并在测试里断言；`tests/scenarios/test_coverage.py` 要求每一项至少有一个测试覆盖。
- 框架代码不写死本机地址、路径或 PID（P7）；超时、心跳之类的可调参数一律做成带默认值的配置项（`doc/specs/README.md`）。
- 框架本体对外返回 `Ok` / `Failed`，不靠异常跨进程边界。
