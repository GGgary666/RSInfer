# CUDA Graph：实现简化与 A/B/C 对抗验证

日期：2026-11-10。

## 对比 master 的实现复审

原功能补丁相对开发基线 `master@efce690`，`engine/` 改动 29 文件、621 行新增、36 行删除。复审期间发现远程 master 已到 `07442d6`，包含状态池释放及重新声明修复；B 合入该版本并保留 graph 的独立 scratch。

按同一最新 master 比较：复审前 `3dc7b1a` 为 +632/-67，复审后为 +615/-46，仍是 29 文件。这里只减少了补丁总改动量，没有显著缩小功能实现；不能把上游修复和文件移动算成大幅精简。

| 部分 | 是否保留及理由 |
|---|---|
| CUDA 平台、生命周期接口、进程接入 | 保留。复用 torch_eager 的张量/stream，新增 capture/replay/close；不另建 executor |
| 固定 attention/GDN 元数据、RoPE、参照 kernel | 保留。消除 capture 中的 CPU 标量读取，防止长度/映射在录制时冻结 |
| KV/GDN scratch、holder 预算、batch 表预分配 | 保留。每 lane 独立 scratch，不能用活跃请求的状态换取少几行代码 |
| 依赖守卫、注册来源、加载/回滚顺序 | 保留。CUDA replay 绕过 Python，服务代理失效本身不足以阻止旧图执行 |
| plan 构建组织 | 简化。删除启动时重复的 backend resolve；将 11 参数的全局 build 函数改为 5 参数的单桶 capture 方法，集合的预算/发布/清理由 plan 单元统一管理 |
| backend 绑定类型 | `NoBuffers` 改名 `GraphBinding`，去掉实际未使用的可选空值，明确其持有固定元数据 |

不新增通用 builder、第二条执行路径或新的跨单元抽象。保留已有 rebuild action 与事务回滚流程，避免为减少 plan 的行数扩展框架协议。

## 实验方法

沿用 session `01a1af53-8a60-7b73-8152-3c74bf6abde5` 中的方法：A 是缺少能力的基线，B 是正确实现，C 从 B 植入违反目标但能通过初始测试的生产改动；加强同一份测试，使 B 通过、A/C 被拒绝。不使用 xfail、skip，也不靠识别测试名称改变行为。

A 使用 `master@07442d6` 的原始 engine。共同测试在 A 上使用已有 eager CUDA 平台，在 B/C 上使用 CUDA Graph 平台；所有版本都必须通过同一个“存在就绪 decode graph”断言。A 的失败应是缺失能力的断言，不能以模块导入失败充数。

两个 C 分别针对：

- eager 替代 replay：启动照常录图并验证，但每次 decode 调用 Python forward；数值可以完全正确，CUDA Graph 加速目标却没有实现。
- 忽略 catalog version：保留 provider 与 binding generation 检查，只删除 kernel 目录版本检查；目录实例未换但注册内容变化时，旧图可能继续使用旧 kernel。

实验提交、测试文件哈希、OSS release/command 和实际结果在完成运行后记录于下文。
