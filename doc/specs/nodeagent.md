# 规格：节点代理 `core.nodeagent`

- 类型：框架本体。每台机器一个独立进程（单机时也有一个）。改了它要重启引擎。v0 用 Python；它是 Rust 化的第一候选。
- 公共约定见 [README.md](README.md)。

## 1. 职责

1. 按进程管家的指令拉起、停止本机进程。
2. 接收 change-set 的内容（按 hash），物化成本机的快照目录；本机进程从那里加载代码。
3. 提供本机进程之间的共享内存区域（给 [ipc.md](ipc.md) 的同机通道用）。
4. 上报本机的健康状况和平台信息。

## 2. 不负责

- 不做决定，只听进程管家和变更事务的。
- 不管 GPU 分配：用哪几张卡写在 `ProcSpec` 里。

## 3. 快照目录

```
<snapshot_root>/<Identity>/        # 只读；一个身份一个目录
    engine/...                     # 引擎源码树
    manifest.json                  # 文件 -> hash
```

- 物化：先按 hash 从控制面的 blob 存储拉缺的文件（本机已有的 hash 直接复用），写到临时目录，校验全部 hash，再原子地改名成 `<Identity>`。
- 目录一旦建好就不再修改。同一个身份在不同机器上的目录内容完全相同。
- `snapshot_root` 必须是本机路径（不要放在 Lustre 这类共享存储上：import 大量小文件会很慢）。
- 清理：保留当前身份、正在准备的身份、最近 N 个身份。N 是配置项 `nodeagent.snapshot_retention`，默认 5（已确认）。

## 4. 进程管理

- `Spawn(ProcSpec)`：用 `ProcSpec` 里的身份对应的快照目录作为代码来源，设好环境变量和 `CUDA_VISIBLE_DEVICES`，拉起进程。
- `Stop(ProcId, timeout)`：先通知进程自行退出，超时后强杀。
- 进程退出时上报退出码。

## 5. 上报

- 心跳：本机每个进程的状态。
- 平台信息：驱动、CUDA、torch 版本、网卡、NVSHMEM 版本（多机时用来检查机器之间是否一致；不一致就拒绝加入）。

## 6. 对外接口（消息）

| 消息 | 方向 | 内容 |
|---|---|---|
| `Register` | 代理 → 管家 | HostId、平台信息 |
| `Materialize` / `Materialized` | 双向 | 身份、文件清单；结果 |
| `Spawn` / `Stop` | 管家 → 代理 | ProcSpec / ProcId |
| `Heartbeat` | 代理 → 管家 | 本机进程状态 |
| `Exited` | 代理 → 管家 | ProcId、退出码 |

## 7. 验收测试（阶段 1）

1. 物化：同一个 change-set 在两个假主机上物化出的目录内容完全相同；中途失败不留下半成品目录。
2. 复用：已有的 hash 不重复下载。
3. 拉起的进程只从快照目录加载引擎代码，不读 agent 的工作区（检查 `sys.path` 和实际 import 的文件路径）。
4. 停止：进程不响应时，超时后被强杀并上报。
5. 平台信息不一致的假主机被拒绝加入。
