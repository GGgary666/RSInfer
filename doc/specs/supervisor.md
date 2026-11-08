# 规格：进程管家 `core.supervisor`

- 类型：框架本体。跑在控制面进程。改了它要重启引擎。v0 用 Python；只通过版本化消息和节点代理交互，以后可以换成 Rust。
- 公共约定见 [README.md](README.md)。

## 1. 职责

1. 持有拓扑数据，按它决定要有哪些进程、各在哪台机器、各装哪些单元。
2. 通过每台机器的节点代理拉起、停止、重启进程；不自己 spawn。
3. 健康检查：进程挂了、整台机器失联，都报告给 `core.change`。
4. 执行 `core.change` 交给它的“重启进程”“改拓扑”动作。

## 2. 不负责

- 不决定什么时候重启（由 `core.change` 决定）。
- 不分配资源：有哪些机器、哪些 GPU，由拓扑数据给定（v0-plan 第 28 条）。

## 3. 拓扑数据

```python
class Topology(msgspec.Struct):
    hosts: list[HostSpec]          # 每台机器：HostId、地址、GPU 列表、本机路径（快照根目录、权重路径）
    mesh: Mesh                     # 并行维度及大小，例如 {"tp": 2} 或 {"dp": 8, "ep": 8}
    placement: list[RankPlacement] # 每个 rank 在哪台机器、本机第几张卡、哪个 NVLink 域
    roles: list[RoleSpec]          # 进程角色：frontend / scheduler / worker / ...
                                   # 每个角色：几个、放在哪台机器、装哪些单元、环境变量、
                                   # 跑在哪个平台（platform、platform_options，2026-11-08）
    rendezvous: str                # 建通信组用的地址（来自数据，不写 localhost）
```

- 单机是 `hosts` 只有一项的特例。
- 拓扑数据进系统身份。
- 改拓扑就是一种 change-set（`topology` 字段）。

## 4. 进程生命周期

```
DECLARED → SPAWNING → RUNNING → STOPPING → STOPPED
                         │
                         └──心跳超时 / 退出──▶ LOST
```

- 拉起：向目标机器的节点代理发 `Spawn(ProcSpec)`，`ProcSpec` 含角色、索引、身份（决定用哪个快照目录）、环境变量、GPU，以及角色指定的平台（[platform.md](platform.md) §2.4）。平台在进程启动时选定，角色换平台就重启它的进程。
- 进程起来以后向控制面报到，带上 `ProcId` 和自己装好的单元列表。
- 报到时带上进程已经打开的数据通道读端地址。之后才打开的读端，用 `PublishEndpoint` 补登记。别的进程用 `Lookup` 查，只查得到 RUNNING 的进程的地址。进程要共享内存环的名字时，进程管家把 `AllocShm` 转给那台机器的节点代理（2026-11-08，[ipc.md](ipc.md) §6）。
- 停止：先发 `Stop`（进程卸载全部单元后退出），超时后让节点代理强杀。
- 重启 = 停止 + 拉起。被重启进程里的状态单元会随之消失，由 `core.change` 事先安排排空。

## 5. 健康检查

- 节点代理每隔固定间隔上报本机进程的状态。间隔是配置项 `nodeagent.heartbeat_interval_s`，默认 1 秒（已确认）；超时判定是配置项 `supervisor.heartbeat_timeout_s`。
- 进程退出或心跳超时：标为 `LOST`，向 `core.change` 报“进程丢失”。
- 节点代理本身失联：这台机器上的所有进程都标为 `LOST`，报“机器丢失”。机器是故障的单位。
- 框架对丢失不自动处理：只报告，并提供“重启进程”“改拓扑”这些动作。等它恢复、重建还是降级，由 RSI agent 或使用者决定（已确认）。

## 6. 对外接口（概要）

```python
class Supervisor:
    def apply_topology(self, topo: Topology) -> Result[TopologyPlan]: ...  # 算出要起、停、重启哪些进程
    def restart(self, procs: list[ProcId]) -> Result[None]: ...
    def status(self) -> list[ProcStatus]: ...
```

## 7. 验收测试（阶段 1）

1. 冷启动：按拓扑数据拉起全部进程，每个进程装上规定的单元。
2. 改拓扑：TP1 → TP2，只重建 rank 进程；前端、调度宿主、控制面不停。
3. 进程丢失：强杀一个 worker，管家在超时内报告“进程丢失”。
4. 机器丢失：停掉一个假主机的节点代理，报告“机器丢失”，那台机器的进程全部标为 `LOST`。
5. 两个假主机：进程按拓扑分布在两台“主机”上，以上全部成立；代码里没有任何 `localhost` 或本机 PID 假设。
