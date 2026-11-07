# 框架本体规格：总览与公共约定

- 日期：2026-11-05（草稿，阶段 0）
- 真源说明：设计以 [../modules.html](../modules.html) 为准；做的顺序见 [../roadmap.md](../roadmap.md)。本目录的规格是对 modules.html 的细化，二者冲突时以 modules.html 为准。
- 标注“（待确认）”的地方是本文自行决定、还没有和发起人确认的。

## 文件

| 文件 | 组件 | 跑在哪里 |
|---|---|---|
| [unit-runtime.md](unit-runtime.md) | 单元运行时 `unit.runtime` | 每个 Python 进程内部 |
| [context.md](context.md) | 配置与运行环境 `core.context` | 每个 Python 进程内部 |
| [ipc.md](ipc.md) | 进程间传输 `core.ipc` | 每个进程内部 + 进程之间 |
| [platform.md](platform.md) | 平台 `platforms.*` | 每个 Python 进程内部 |
| [nodeagent.md](nodeagent.md) | 节点代理 `core.nodeagent` | 每台机器一个独立进程 |
| [supervisor.md](supervisor.md) | 进程管家 `core.supervisor` | 控制面进程 |
| [change.md](change.md) | 变更事务 `core.change` | 控制面进程 |
| [control-plane.md](control-plane.md) | 控制面进程 `core.control`，以及控制面单元 `core.identity`、`core.manifest`、`core.deps`、`core.api` | 控制面进程 |
| [state-changes.md](state-changes.md) | 状态单元的变更：排空与版本迁移（只有规格，实现在阶段 2） | 调度宿主、worker、控制面 |

前四个和单元在同一个 Python 进程里，只能用 Python；后三个是独立进程，只通过版本化消息交互，以后可以单独换成 Rust（v0 全部 Python）。

## 公共约定

### 语言和运行环境

- Python 3.12 及以上（已确认）。理由：`sys.monitoring` 用来低开销地记录“录 graph 时执行过哪些代码”。
- 控制面、节点代理、调度宿主的事件循环用 `asyncio`（已确认）；worker 的每步循环是同步的单线程循环，框架的重载和状态动作只在 epoch 边界、在这个线程里执行。
- 跨进程的消息用 `msgspec.Struct` 定义（sglang、mini-sglang 都在用），编码用 msgpack。msgspec 的结构可以一一对应到 Rust 的 serde，方便以后换语言。

### 标识

| 名字 | 含义 | 形式 |
|---|---|---|
| `HostId` | 一台机器 | 拓扑数据里的字符串，例如 `"host-a"` |
| `ProcId` | 一个进程 | `(HostId, role, index)`，例如 `("host-a", "worker", 3)` |
| `Endpoint` | 一个可寻址的通信端点 | `(ProcId, channel_name)` |
| `UnitId` | 一个单元（代码层面） | 清单里的 `id`，例如 `"policies.chunk_fcfs"` |
| `InstanceId` | 单元在某个进程里的一次加载 | `(ProcId, UnitId, generation)`；每重载一次 `generation` 加 1 |
| `Identity` | 系统身份 | 身份清单的 hash（见 [change.md](change.md)） |
| `Epoch` | 步的编号 | 单调递增的整数；变更只在两个 epoch 之间生效 |

所有标识都不含本机 PID、本机路径或 `localhost`（P7）。

### 单元清单（`unit.toml`）

每个单元目录下一份（已确认用 `unit.toml`，不用 Python 装饰器：静态检查和身份计算不用 import 代码就能读）。字段的细节在阶段 0 评审时定：

```toml
[unit]
id = "policies.chunk_fcfs"
kind = "logic"                      # logic | state；框架本体不是单元，没有清单
process = ["scheduler"]             # 装在哪些进程角色里
entry = "policies.chunk_fcfs:Unit"  # 模块:类

[provides]
services = ["sched.chunk_policy@1"] # 服务名@接口版本

[requires]
services = ["sched.snapshot@1"]     # 必须有；缺了就停在 PENDING
optional = []                       # 可选；用 ctx.get() 取，可能为 None

[config]                            # 只能读声明过的键
"sched.chunk.max_tokens" = { type = "int", default = 8192, binding = "live" }  # live | plan | startup

[capabilities]
accept_config = true                # 能原地接受新配置
actions = []                        # 状态单元才有：refit | grow | migrate | invalidate
graph_capturable = true             # kernel solution 才有意义
native_extensions = []              # 用到的 .so；改了它们走“重启进程”
changes_kv_numerics = false         # 模型、层单元才有意义

[state]                             # 只有状态单元有
drain_on_reload = true              # 默认 true：重载前排空
migrate = ""                        # 提供迁移函数时写在这里，例如 "mod:migrate_v1_to_v2"
```

清单是声明；运行时和录 graph 时自动记录的依赖用来交叉校验。清单少报、多报都会被 `core.manifest` 发现。

### 单元的编写接口（概要）

完整接口见 [unit-runtime.md](unit-runtime.md)。一个单元是一个类：

```python
class Unit:
    def __init__(self, ctx: UnitContext, config: Config) -> None: ...
    # 可选：装好以后、开始服务之前调用
    def start(self) -> None: ...
```

- 所有注册都通过 `ctx` 完成，每一项都自动登记撤销函数：`ctx.provide(name, obj)`、`ctx.effect(setup)`、`ctx.register(registry_service, item)`、`ctx.on_config(key, handler)`。
- 取别的服务只能用 `ctx.use(name)`（必须有）或 `ctx.get(name)`（可选），不许 import 别的单元的对象。
- 单元的 `__init__` 和撤销函数都不许阻塞太久；要做的长时间工作放进 `prepare` 钩子（见 [change.md](change.md)）。

### 消息信封

所有跨进程消息都套一层信封：

```python
class Envelope(msgspec.Struct):
    schema: str          # 消息类型名@版本，例如 "sched.step_plan@1"
    sender: Endpoint
    epoch: int | None
    body: bytes          # msgpack 编码的消息体
```

收到不认识的 `schema` 或版本对不上，直接拒收并报诊断，不猜。

### 可调参数

框架本体里的超时、心跳间隔、保留数这类可调参数一律是配置项（带默认值，名字形如 `组件.参数`），不写死在代码里。本目录各规格里出现的默认值都是这类配置项的默认值。

### 结果和错误

- 框架本体对外的操作都返回结果对象：`Ok(value)` 或 `Failed(diagnostics)`。不靠异常跨越进程边界。
- `Diagnostics` 至少包含：哪个组件、哪个阶段、哪个单元或进程、原因、可操作的建议。
- 任何“推导不出来”的情况一律保守处理：升级到更贵的动作，并在诊断里说明。

### 测试约定

每个组件的规格最后都有“验收测试”一节，列出它在阶段 1 必须通过的测试。所有测试在 CPU 上、用 fake 平台跑；涉及多机的测试在一台机器上起两个节点代理模拟。
