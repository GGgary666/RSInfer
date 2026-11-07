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
  - 文档里的路径（`README.md`、`doc/`、`tests/framework/formal/README.md`）和本文件的目录地图。

## 目录地图

### 顶层

| 路径 | 内容 |
|---|---|
| `engine/` | 引擎源码树（框架本体 + 将来的单元） |
| `tests/` | 全部测试 |
| `doc/` | 设计文档、规格、评估记录 |
| `.github/workflows/ci.yml` | CI：`uv sync --python 3.12`、`uv run pyright`、`uv run pytest`（Ubuntu） |
| `.github/workflows/notify.yml`、`.github/scripts/notify.py` | PR 合进 `master` 后由 LLM 判断是否值得通知，值得就发一句话摘要到通知群；密钥 `PR_NOTIFY_WEBHOOK`、`PR_NOTIFY_SECRET`、`LLM_API_KEY`、`LLM_BASE_URL`，可选变量 `LLM_MODEL` |
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
| `engine/unit/runtime/` | 单元运行时：加载 / 重载 / 撤销、`ProcessHost`、注册表、静态 import 图 | `unit-runtime.md` |
| `engine/platforms/` | 平台接口 `base.py`；CI 用的 `fake/` | `platform.md` |

将来的单元（调度策略、KV cache、模型等）也放在 `engine/` 下，但它们不属于框架本体，不要列进 `framework.toml`。

### `tests/`

- `tests/__init__.py`：让 `tests` 成为包。所有测试辅助代码都用 `tests.framework.…` 绝对导入。
- `tests/framework/`：框架本体（`engine/framework.toml` 列出的模块）的测试。
  - `conftest.py`：autouse 的 `_isolate_imports`（每个测试后恢复 `sys.path` / `sys.modules`，因为 toy unit 被导入进全进程共享的 `sys.modules`），以及 `pkg`、`snapshot` fixture。
  - `support/`：通用工具 `spec`、`manifest_toml`、`write_tree`、`PROC`、`events`、`GENERIC_UNIT`。
  - `unit_runtime/`、`context/`、`ipc/`、`platforms/`、`change/`、`cluster/`：按组件划分。`ipc/test_location_transparency.py` 扫描整个 `engine/`，禁止写死本机地址或用 PID 当地址（P7）。
  - `adversarial/`：对抗与故障边界测试，对应 `doc/evaluations/` 的记录。
  - `formal/`：TLA+ 有限模型（`models/`）、runner（`run.py`）、对真实实现的轨迹回放和契约测试。细节见 `tests/framework/formal/README.md`。
- 测试辅助模块（不以 `test_` 开头，pytest 不收集）：

  | 模块 | 提供 |
  |---|---|
  | `framework/cluster/harness.py` | 测试事件循环里的真实控制面 + 子进程里的节点代理（`Cluster`，`async with cluster(...)`） |
  | `framework/change/engine.py` | 基于 cluster harness 的协调者测试台（`Engine`，`async with engine(...)`） |
  | `framework/change/harness.py` | 进程内 Coordinator + 两个真实 `ProcessHost`，可在传输边界丢请求 / 回执（`Harness`、`Capture`、`OLD`、`harness`） |
  | `framework/adversarial/peer.py` | 单个真实 `ProcessHost`，逐条投递参与者消息（`Peer`、`peer`） |
  | `framework/ipc/peers.py` | IPC 测试在子进程中运行的代码 |

- 被多个测试文件共用的代码放进上面这类辅助模块（同目录或 `support/`），**不要从另一个 `test_*.py` 里导入**。fixture 以 `from … import harness as harness` 的形式在测试文件中重新导出。

### `doc/`

- `doc/modules.html`：**设计真源**。冲突时：`modules.html` > `v0-plan.md` > `discussion-summary.md`。
- `doc/roadmap.md`：阶段和验收。
- `doc/specs/`：框架本体七个组件的规格；`README.md` 是总览和公共约定（标识、清单、信封、结果、测试约定）。
- `doc/v0-plan.md`、`doc/discussion-summary.md`：v0 计划、讨论纪要。
- `doc/evaluations/`：历次修复 / 对抗测试的评估记录。里面的提交号、SHA-256、通过数是历史数据，不要改；路径变动时只更新命令。
- `doc/archive/`：旧版文档，不再维护。
- `doc/figures/`：图片。

## 常用命令

```sh
uv sync --python 3.12
uv run pyright                       # strict；检查 engine/ 和 tests/（含 formal）
uv run pytest                        # 默认测试；不收集 tests/framework/formal
uv run pytest tests/framework/ipc -q # 只跑某个目录
# formal 全流程：需要 Java 11+；首次 --bootstrap 下载并校验 TLC 1.7.4
uv run python tests/framework/formal/run.py --bootstrap --replay [--java /path/to/java]
```

- 每次提交前跑 `uv run pyright` 和 `uv run pytest`。改了 `core.change`、`core.supervisor`、`core.nodeagent`、`unit.runtime.host` 或 `tests/framework/formal/` 时，也跑 formal 全流程。
- `uv run pytest` 跳过 formal 靠的是 `pyproject.toml` 里 `addopts` 的 `--ignore=tests/framework/formal`；显式给出路径（`uv run pytest tests/framework/formal`）不受影响。
- 生成物不进仓库：`tests/framework/formal/.tools/`、`.runs/` 已被忽略。

已知与环境有关的失败：`ipc/test_streams.py::test_slow_consumer_blocks_the_producer[tcp]` 在 macOS 上偶发失败，Linux 和 CI 上正常。

## 约定

- 文档用中文；代码、注释、docstring 用英文。
- 测试不用 `xfail`，也不按平台或缺少工具自动 `skip`。
- 框架代码不写死本机地址、路径或 PID（P7）；超时、心跳之类的可调参数一律做成带默认值的配置项（`doc/specs/README.md`）。
- 框架本体对外返回 `Ok` / `Failed`，不靠异常跨进程边界。
