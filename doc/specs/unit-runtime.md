# 规格：单元运行时 `unit.runtime`

- 类型：框架本体。跑在每个 Python 进程内部。改了它要重启引擎。
- 借鉴：Cordis 的 Fiber / Service / effect / HMR（deepseek-harness `vendor/cordis/src/fiber.ts`、`vendor/hmr/src/index.ts`）。
- 公共约定见 [README.md](README.md)。

## 1. 职责

1. 按进程计划装单元：这个进程装哪些单元、各自的配置。
2. 管单元的生命周期和服务注册表。
3. 管可撤销注册：卸载时倒着执行撤销函数。
4. 依赖传播：提供者换了，依赖它的单元自动重载。
5. 代码重载：从本机快照目录重新 import 被改的模块，卸旧装新，失败回滚。
6. 保证卸载顺序：先卸使用方，再卸提供方。

## 2. 不负责

- 不决定重载什么（由 `core.deps` 推导，`core.change` 下指令）。
- 不决定什么时候重载（只在调用方给的“安全点”执行；worker 的安全点是 epoch 边界）。
- 不认识任何领域概念（调度、KV、TP 都不认识）。

## 3. 生命周期

每个单元实例的状态：

```
PENDING ──依赖都齐了──▶ LOADING ──__init__ 成功──▶ ACTIVE ──卸载──▶ UNLOADING ──▶ DISPOSED
   ▲                        │                          │
   └────依赖消失──────────────┴──失败──▶ FAILED            └──依赖的提供者换了──▶ UNLOADING ──▶ PENDING
```

- **PENDING**：缺必须的服务。不报错，等。
- **FAILED**：`__init__` 或 `start` 抛错。已经登记的撤销函数全部执行，状态回到干净；诊断上报。
- 每个实例记一个**依赖指纹**：它用到的每个服务当前提供者的 `InstanceId` 拼起来。指纹变了就卸载再装（Cordis 的 epoch 机制）。

## 4. 服务注册表

- `ctx.provide(name, obj)`：登记服务。名字带接口版本，例如 `"sched.chunk_policy@1"`。同一个进程里一个名字只能有一个提供者。v0 不支持 Cordis 的 isolate（已确认）；shadow / canary 绑定以后再加。
- `ctx.use(name)`：取必须有的服务；只能取清单 `requires` 里写了的。
- `ctx.get(name)`：取可选服务；可能返回 `None`。
- 取到的对象是**代理**：提供者被重载后，旧代理再被调用就报错。这样“留着旧对象”的错误能被当场发现。

## 5. 可撤销注册

- `ctx.effect(setup)`：`setup()` 返回一个撤销函数。
- `ctx.register(registry, item)`：向另一个单元提供的注册表登记一项，注册表必须返回撤销函数。
- `ctx.on_config(key, handler)`：原地接受新配置的钩子（见 [context.md](context.md)）。
- 撤销函数按登记的**相反顺序**执行；一个撤销函数失败，记录诊断，继续执行其余的。
- 卸载完成后检查：这个实例登记过的所有东西都撤掉了。没撤掉就是缺陷（植入缺陷测试会用到）。

## 6. 代码重载

### 6.1 输入

来自 `core.change` 的重载指令：`ReloadPlan(changed_modules, units_to_reload, units_to_add, units_to_remove, new_snapshot_dir)`。

### 6.2 步骤

1. **确认安全点**：调用方保证当前没有在执行的步。
2. **算卸载集合**：要重载、要删除的单元，加上依赖它们的单元（传递闭包）。
3. **按依赖反方向卸载**：使用方先卸，提供方后卸。
4. **备份并清掉模块缓存**：被改的模块，以及 import 了它们的模块，从 `sys.modules` 里移走并备份。
5. **切换代码来源**：`sys.path` 里引擎代码的那一项指向新的快照目录（快照目录由节点代理物化，见 [nodeagent.md](nodeagent.md)）。
6. **按依赖顺序装载**：提供方先装，使用方后装。
7. **出错就回滚**：恢复模块缓存和代码来源，按原来的配置重新装回旧单元；把回滚结果和诊断交回 `core.change`。

### 6.3 新旧代码不能共存

同一个模块名在一个进程里只能有一个版本。所以 v0 规定（已确认）：
- **新增的单元**（新模块）可以在准备阶段就后台装好，和旧单元并存。例如“新增一个 kernel solution 单元，后台用它重录 graph”。
  - **例外**：新单元直接或间接 import 了**任何**改动过的模块，就不能在准备阶段装，只能等安全点。不管这个改动过的模块当前进程有没有加载：没加载的模块会先从旧快照里找到，新单元就会拿到旧代码。
  - 到了安全点，如果发现某个准备阶段装上的单元 import 了改动过的模块（规则被绕过了），直接报错并回滚，不能静默跳过。
- **修改已有模块**时，新代码在 epoch 边界的暂停期内才加载；依赖它的派生产物（graph）也在暂停期内重建。停顿更长，但正确。
- 以后可以用“每个身份一个包名前缀”让新旧版本并存，留作优化。

### 6.4 改不了的东西

- 清单里声明的 native 扩展（.so）变了：不重载，直接返回“需要重启进程”。
- 框架本体的模块变了：返回“需要重启引擎”。
- 发现未声明的 native 模块被改：同样返回“需要重启进程”，并报清单缺陷。

## 7. 对外接口（概要）

```python
class UnitRuntime:
    def load_process_plan(self, plan: ProcessPlan) -> Result[None]: ...
    def reload(self, plan: ReloadPlan) -> Result[ReloadReport]: ...       # 只在安全点调用
    def stage_new_units(self, units: list[UnitSpec]) -> Result[None]: ... # 准备阶段：后台装新增单元
    def stage_in_slices(self, units: list[UnitSpec]) -> Generator[None]: ...  # 同上，每次 next() 装一个（2026-11-07）
    def snapshot(self) -> RuntimeSnapshot: ...  # 每个实例的状态、依赖指纹、登记过的东西
```

`RuntimeSnapshot` 也是自省的数据来源。

## 8. 验收测试（阶段 1）

1. 生命周期：缺依赖停在 PENDING；补上后自动 ACTIVE；`__init__` 抛错进入 FAILED 且无残留。
2. 撤销：随机生成单元和注册，卸载后注册表和装载前完全一致。
3. 依赖传播：重载提供者，所有依赖它的单元都被重载，不依赖它的都没动。
4. 卸载顺序：随机依赖图下，卸载顺序总是先使用方后提供方。
5. 旧代理：重载后调用旧代理当场报错。
6. 回滚：在第 6 步任意位置注入失败，回滚后模块缓存、注册表、单元状态都和重载前一致。
7. native 扩展：改了声明过的 .so，返回“需要重启进程”，不做任何卸载。
8. 新增单元可以在准备阶段装好；修改已有模块只在安全点生效。
9. 准备阶段的例外：新单元 import 了一个改动过、但当前进程**还没加载**的模块，准备阶段拒绝装它；改到安全点装以后，它用的是新代码。
