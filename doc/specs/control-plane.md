# 控制面：控制面进程和四个控制面单元

- 日期：2026-11-07（阶段 1 第二块的设计和实现笔记，边做边记）
- 对应：[../roadmap.md](../roadmap.md) 阶段 1 的“控制面单元”和三类验收测试；modules.html 里的 `core.api`、`core.identity`、`core.manifest`、`core.deps`。
- 发起人已确认（2026-11-07）：
  1. 控制面单元装在控制面进程里。改它们的变更，由协调者在本进程执行（它一次只处理一个变更，两次变更之间就是它的安全点），失败就整体撤回；改框架本体仍然是重启引擎。
  2. 玩具单元只在“非常简单、对人理解代码有必要”时才写，不做系统模拟器。场景测试用到的单元是测试夹具，放在 `tests/scenarios/`，不进 `engine/`。
  3. 控制面单元放 `engine/core/{identity,manifest,deps,api}/`（单元 id 就是 `core.*`，但它们是单元，不进 `framework.toml`）；它们的测试放 `tests/units/`；场景测试放 `tests/scenarios/`。

## 1. 控制面进程（框架本体，`core.control`）

入口：

```sh
python -m core.control serve --source <工作区> --topology <topology.json> --state-dir <目录> \
    --host <HostId> --bind <地址> [--config <config.json>] [--spawn-agents]
```

- 持有：blob 存储、事件日志、进程管家、协调者，以及一个单元运行时，装 `process` 里含 `"control"` 的单元（就是四个控制面单元）。这个进程的 ProcId 是 `(host, "control", 0)`。
- `--spawn-agents`：给拓扑里的每台主机在本机起一个节点代理（单机就是多机的特例；地址来自拓扑数据，P7）。多机部署时不加，各机器自己起节点代理。
- **启动顺序**：
  1. 把工作区 `engine/` 下的文件收进 blob 存储；
  2. 按文件清单的 hash 在本机物化一份快照，控制面单元从这里加载（和别的进程一样，不从工作区直接 import）；
  3. 读快照里所有 `unit.toml`，装上 `process` 含 `"control"` 的单元；
  4. 提交启动变更（从空系统到第一个身份）：用 `core.manifest` 检查、`core.deps` 出计划、`core.identity` 算身份，进程管家按计划拉起进程。
- **宿主服务**：控制面单元需要框架的东西（blob 存储、进程管家、协调者、事件日志），由宿主以服务的形式提供：`ctrl.blobs@1`、`ctrl.supervisor@1`、`ctrl.change@1`、`ctrl.events@1`。宿主服务不属于任何单元，不会被卸载（`UnitRuntime.provide_host`）。
- **协调者的三个接口**改成每次变更时向服务要：`ctrl.manifest@1`（静态检查）、`ctrl.deps@1`（动作计划）、`ctrl.identity@1`（身份）。服务不在（单元没装上或者失败了）就 `Failed(control.unit_missing)`，什么都不动。它们来自单元，抛异常时协调者把它变成 `change.check_crashed` / `change.plan_crashed`；变更记录一旦开始，无论哪里抛异常，都以 `failed` 结束，不会停在 `received`。
- **控制端口**：沿用进程管家已有的控制端点。`core.control.protocol` 定义 API 消息，宿主把它们转给 `ctrl.api@1`（`core.api` 单元）。客户端：`python -m core.control.client --control <地址> status|apply|check|introspect|logs`。
- **控制面单元的重载**：动作计划里多一段 `ActionPlan.control`（给控制面进程的 `ProcActions`）。协调者在 Execute 阶段、所有参与者都执行完之后在本进程执行它；撤回时按反向计划恢复。
  - 执行完、提交前检查：四个控制面单元都 ACTIVE，而且 `ctrl.identity@1`、`ctrl.manifest@1`、`ctrl.deps@1`、`ctrl.api@1` 都有人提供（和启动时同一个检查）。不满足就撤回，否则提交以后控制接口就没了。
  - 撤回没能把旧的控制面单元装回来（反向重载失败、装回来以后服务不全），和远端参与者没确认撤回一样：协调者进入“需要恢复”，拒绝后续变更。
- **控制面单元从哪里加载**：控制面进程自己的框架代码来自启动它的地方（工作区或仓库），控制面单元来自快照。为此单元运行时装进程计划时，会让已经导入的包先去快照目录里找还没加载的模块（`UnitRuntime._prefer_root`）；从快照启动的进程里这一步什么都不做。
  - 这些包（比如 `core`）的 `__file__` 不在快照里，但 `__path__` 里有快照目录。重载换快照时，切换、备份、恢复搜索路径的包，除了从快照加载的，也包括 `__path__` 里有快照目录的（2026-11-07 复审修复：原来只看 `__file__`，`core.__path__` 一直指着第一个快照，改了 `core/sourcetree.py`、在 `core/` 下加控制面单元、改控制面单元包的 `__init__.py`，提交了却还在跑旧代码，或者找不到新模块）。
  - 第一次装进程计划时，已经从别处导入、快照里也有的非框架模块（单元代码、库）会先从 `sys.modules` 里去掉，再从快照导入：单元代码只从快照跑。单独起的控制面这时只导入了框架代码，这一步什么都不做；在测试进程里跑的控制面可能已经导入了单元模块（比如 `tests/units` 先导入了 `core.sourcetree`），以前会一直用仓库里那份。

## 2. `core.identity`：系统身份

- 身份 = 对“身份清单”算 hash：文件内容（含各单元的 `unit.toml`）、配置、依赖锁、产物（权重清单等）、拓扑数据，再加上平台信息（各机器一致，进程管家在注册时已经检查）。
- 保证 `候选身份 = f(起点身份对应的清单, change-set)` 是确定的：清单按键排序后编码，和 change-set 里条目的顺序无关。
- 给出两个身份之间的差异（`diff`）。
- 框架里 `core/change/state.py` 的默认实现保留，给不起控制面的测试用。

## 3. `core.manifest`：清单与静态检查

只读文件，不运行代码。检查失败就 `Failed`，什么都不动。

| 检查 | 诊断码 |
|---|---|
| 每个 `unit.toml` 能解析，单元 id 不重复，入口模块存在 | `manifest.invalid`、`manifest.duplicate_unit`、`manifest.missing_entry` |
| 每个 `.py` 都能解析（报文件和行号）；解析不了的模块，后面读代码的检查跳过它 | `manifest.syntax_error` |
| 每个配置键只有一个主人；`[reads]` 里的键必须有人声明 | `manifest.config_owner`、`manifest.undeclared_read` |
| 代码里 `config["k"]`、`config.get("k")`、`config.peek("k")` 读到的键必须在清单里声明；声明了却从没读过的报警告 | `manifest.read_not_declared`、`manifest.unused_read`（警告） |
| 拓扑里每个进程角色：每个单元的必需服务都有人提供；单元的 `process` 含这个角色 | `manifest.unprovided_service`、`manifest.wrong_role` |
| 单元的代码不 import 别的单元目录里的模块（只能通过服务名拿） | `manifest.cross_unit_import` |
| 单元模块里没有模块级可变全局变量，也不用 `global` | `manifest.mutable_global` |
| 只有声明了 `captures_graph` 的单元调用 `platform.graph()` | `manifest.graph_outside_plan` |
| 单元目录里的 `.so` 必须声明在 `native_extensions` 里 | `manifest.undeclared_native` |

## 4. `core.deps`：影响分析

输入：change-set、起点和候选的身份清单、进程管家的进程表，以及向每个在跑的进程要来的自省信息（装了哪些单元、配置在哪个阶段被谁读了、登记了哪些派生产物）。输出：动作计划。`apply --dry-run` 返回的就是它。规则对应动作阶梯：

| 改了什么 | 动作 | 阶梯 |
|---|---|---|
| 框架本体的模块 | `restart_engine` | 6 |
| 单元声明的 native 扩展 | 重启装了这个单元的进程（有排空单元就先排空） | 5 |
| 拓扑的 mesh / 放置 / 主机 / GPU / 环境 | 重启受影响的 rank 进程；新增的起、去掉的停 | 5 |
| 状态单元管的内容（内部变更里的状态请求） | 状态单元自己的动作（如 `migrate`） | 4 |
| 被派生产物按值依赖的配置键或单元代码 | 产物主人的重建动作（准备阶段在第二套 buffer 里做） | 3 |
| 单元代码或清单 | 重载这个单元（依赖它的单元由运行时跟着重载） | 2 |
| 新增单元 | 准备阶段装上，前提是它建起来之前要的东西那时都已就绪（见下）；否则到安全点随重载一起装 | 0 / 2 |
| 读它的单元注册了钩子（`ctx.on_config`），键不是 `startup` 绑定 | 原地接受新值 | 1 |
| 读它的单元没注册钩子，或者键是 `startup` 绑定 | 重载读它的单元 | 2 |

- 每个进程的单元：拓扑里角色的 `units` 不为空就用它；为空就是所有 `process` 含这个角色的单元。改单元列表在原地加、减单元，不重启进程。
- “改动过的模块”只算原来就有、被改了或删了的模块；新文件不算（新单元是加进来，不是重载）。
- 新单元能不能在准备阶段装：它建起来之前，代码、配置声明、候选值和要用的服务都必须已经就绪。下面任何一条不满足，就到安全点随重载一起装（重载先声明新键、应用新值，再建单元），并在 `notes` 里写明原因：
  - 不 import 改动过的模块；
  - 它读的键在这个进程里已经声明，而且这次变更不改它的值；或者键的主人也在准备阶段装。读别的角色的单元拥有的键，属于前者不满足、后者也不满足的情况；
  - 它自己拥有的键没被进程里别的单元声明着（键从要删的单元转给它时，要等重载先删掉旧主人）；
  - 它要用的服务（必需的和可选的）已经有人提供，或者由同样在准备阶段装的新单元提供；
  - 推迟一个单元，可能让依赖它的单元也推迟，所以反复检查到不再变化。
- 进程计划里的配置项（启动、重启、新起的进程，和控制面启动时）都由同一个函数算：单元拥有的键和它 `[reads]` 里别的单元拥有的键，值是给定的或主人的默认值（`core.context.config.config_entries`）。参与者把变更的值分成两份：新单元拥有的键跟着新单元一起声明，其余的在重载里生效（`core.change.participant.new_unit_config`，远端参与者和控制面共用）。
- 配置值和重载一起发生时，新值在重载里面生效：重载失败，旧单元按旧值重建。
- 规划前同步问每个在跑的进程要自省信息；有一个进程问不到，整个计划失败（否则它会留在旧身份上）。
- 同样的输入总是给出同样的计划，所以 `--dry-run` 给出的就是实际执行的计划；dry-run 的结果也带着候选身份。
- 推导不出来就保守：升级到更贵的动作，并在 `notes` 里说明。

## 5. `core.api`：控制接口

- `status`：当前身份、状态版本、是否需要恢复、进程表。
- `check`：只做静态检查（场景⑥“不过就直接拒绝，不占 GPU”）。
- `apply [--dry-run]`：客户端先问控制面缺哪些 blob、只上传缺的，再提交文件清单；`core.api` 算出和当前清单的差异，组成 change-set 交给协调者。
- 命令行客户端的退出码：控制面答复了但变更没提交（`failed`、`aborted`、`dropped`）也是 1；只有 `committed`，或者 dry-run 的 `planned`，是 0。
- `introspect`：每个进程的单元、读取记录、派生产物。
- `logs`：事件日志里的变更记录。
- 单元自己发起的内部变更（如 EPLB）：进程宿主给单元提供宿主服务 `proc.control@1`，它把 change-set 发到控制面，由 `core.api` 放进队列。

## 6. 框架本体的改动

- `UnitRuntime.provide_host(name, obj)`：宿主服务。
- `ctx.derived(name, rebuild=..., units=[...])`：登记派生产物；它按值依赖的配置键由 `plan_scope(name)` 里的读取自动记下。
- 进程宿主回答 `Introspect`；提供 `proc.control@1`。
- `ChangeSet.state`：内部变更的状态请求（单元、动作、参数）。
- `RoleSpec.leads_epochs`：哪个角色决定 epoch（调度宿主）。
- `ActionPlan.control`：给控制面进程的动作。

## 7. 场景测试

9 个场景都用测试夹具单元在两台假主机上跑。只测场景依赖的框架机制，不模拟模型、分词、HTTP 这些领域内容。每个场景测到什么、没测什么，见 `tests/scenarios/` 里各测试的 docstring。

## 8. 覆盖检查和拆除检查

- 覆盖：每个测试用 `pytest.mark.covers(edges=..., ladder=..., kinds=...)` 声明自己走到的边（6 种）、动作阶梯的级（0–6）、模块种类（逻辑单元、状态单元、框架本体）；场景测试同时断言实际执行的动作级别。覆盖检查在收集阶段确认三者都至少被一个测试走到。
- 拆除：`engine/` 里没有任何模块 import `rsi`，引擎不装 `rsi/` 也能启动、`apply`。

## 9. 暂不做和已知限制

- **排空**：现在没有单元持有在途请求，`core.deps` 不生成排空动作；场景⑨重启进程时也不排空。
- **状态单元的代码改了**：直接重载，状态丢失；清单里的 `[state] migrate` 还不支持。
- **控制面进程从快照启动**：现在控制面的框架代码来自启动它的地方。以后可以先物化，再从快照 exec 自己，和别的进程一样。
- **控制面单元的动作**：`ActionPlan.control` 只支持重载、增删单元和改配置值，不支持准备阶段装单元和单元动作。
- **事件流**：`core.api` 还没有对外发布事件流，先用 `logs` 轮询。
- **快照清理**：`core.manifest`、`core.deps` 分析候选时在本机物化的快照（按文件清单的 hash 缓存）不清理。
- **状态表里的单元**：`status` 里每个进程的单元列表来自它报到时，原地重载以后不更新；看最新的用 `introspect`。
- **场景测试没有覆盖的**：
  - ① 不含调度宿主和 worker 之间的每步记录流（数据边由 `tests/framework/ipc` 覆盖）；
  - ⑤ 不统计负载、不做 all-reduce、不在 rank 之间搬专家权重；
  - ⑥ 只做引擎这一侧，没有评测服务和谱系（阶段 4）。没有另起一个引擎从“起点 + patch”冷启动来比较身份：身份只取决于清单的内容，这一点由 `tests/units/core/test_identity.py` 的性质测试覆盖。
