# 规格：配置与运行环境 `core.context`

- 类型：框架本体。跑在每个 Python 进程内部。改了它的代码要重启引擎；**改配置的值从来不重启引擎**。
- 借鉴：sglang `runtime_context.py` 的分层（配置 / 运行时标志 / 资源 / 每次前向）。
- 公共约定见 [README.md](README.md)。

## 1. 职责

1. 声明过的配置键的唯一读取入口；读没声明的键直接报错。
2. 运行时标志：不是配置、但运行时会变、而且会影响结果的值（例如 DeepEP 的模式）。
3. 资源租约：CUDA stream、持久 buffer 等，作为可撤销注册发给单元。
4. 记录每一次读取：谁读的、在什么时候读的（每步现读，还是编 plan 时读），交给依赖分析。
5. 配置值变了时，通知读它的单元。

## 2. 不负责

- 不限制环境变量（v0-plan 第 29 条）。
- 不判断改了配置要做什么动作（由 `core.deps` 根据读取记录推导）。

## 3. 配置键

- 每个键在单元清单里声明：类型、默认值、**绑定方式**：
  - `live`：每步现读。改了以后，读它的单元在下一个 epoch 拿到新值；单元可以用 `ctx.on_config(key, handler)` 原地接受。
  - `plan`：编 plan 时读，会被录进 graph。改了以后，用到它的 plan 要重建。
  - `startup`：只在单元装载时读。改了以后，重载读它的单元。
- 键名全局唯一；同一个键可以被多个单元读，但只有一个“所有者”单元声明默认值（待确认）。
- 严格解析：类型不对就报错，不静默回落到默认值。

## 4. 读取记录

```python
class ConfigRead(msgspec.Struct):
    key: str
    reader: InstanceId
    phase: Literal["runtime", "plan_build", "startup"]
    value_hash: str
```

- 运行时的读取只记“谁读过哪些键”，不逐次记录（开销）。
- 编 plan 时的读取逐个记录，并归到当时正在编的那个 plan 上（由 plan 单元开启记录范围）。

## 5. 运行时标志和资源

- 运行时标志只能通过 `ctx.flags` 读写；在录 graph 期间读到的标志值作为“按值”依赖记录下来，replay 前由 plan 单元恢复。
- 资源租约：`ctx.lease_stream(name)`、`ctx.lease_buffer(name, factory)`。名字按用途起（例如 `"alt"`、`"offload"`），不同用途不共用同一个 stream。租约随单元卸载自动归还。

## 6. 验收测试（阶段 1）

1. 读没声明的键报错；类型不对报错。
2. `live` 键：改值后下一个 epoch 生效，不重载任何单元；没有注册钩子的单元被重载。
3. `plan` 键：改值后，依赖分析给出“重建用到它的 plan”；假 CUDA graph 在没重录时 replay 会报错。
4. `startup` 键：改值后只重载读它的单元。
5. 读取记录和清单声明对不上（读了没声明的、声明了从没读的）时，`core.manifest` 能报出来。
