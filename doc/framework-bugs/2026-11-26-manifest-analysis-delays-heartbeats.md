# 控制面静态分析期间节点心跳超时

- 日期 2026-11-26
- 状态 已修，四层 TP2 默认心跳配置下冷启动和热更通过
- 引擎 GLM-5.3-Flash 原生四层 TP2 和完整 TP8
- 操作 冷启动，或在健康的四层实例上用 `control apply --source` 替换 gated RMSNorm 实现
- 现象 冷启动返回 `supervisor.no_host`。热更在 5.983 秒的静态检查和规划阶段后返回四个 `supervisor.no_endpoint`，进程仍留在本机
- 分析 控制面在同一 asyncio 事件循环中处理心跳和候选快照的 AST 分析。时长超过 5 秒的同步检查会推迟心跳请求和超时检查
- 修复 把清单检查、树解析和依赖计划的纯计算移到后台线程，继续在控制面事件循环中完成快照引用和进程 RPC
- 绕过 `serve --control-tunables '{"supervisor.heartbeat_timeout_s":30}'`。采用该配置后 TP8 完整模型启动成功

复测从已经使用过的运行目录启动。热更的静态检查和规划耗时 5.554 秒，事务成功提交。变更只重载两张卡上的 `layers.glm53`，frontend、scheduler 和两张卡的 PID 均保持原值。四条不同长度请求的生成文本一致。
