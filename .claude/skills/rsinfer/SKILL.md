---
name: rsinfer
description: 定位 RSInfer 的引擎单元、配方与测试，在原生运行实例上检查和应用变更、加临时探针、换权重或恢复失败事务。Ascend 910C / CANN 的部署、性能、算子和故障定位从本入口选用对应 skill。
---

# RSInfer 目录地图与热改开发

## Ascend 910C 开发与验证

| 要解决的问题 | 从这里开始 |
| --- | --- |
| 配 CANN / OPP、选物理 die，启动 Qwen3 或 GLM、核对拓扑、跨重启恢复 MemCache | [环境与运行](../rsinfer-ascend-runtime/SKILL.md) |
| 估算 SOL、分析已有 trace、采集性能或用 AISBench 对比 TTFT / TPOT 与吞吐 | [推理性能优化](../rsinfer-ascend-optimize/SKILL.md) |
| 选择 CANN 算子、改 Ascend C tiling 或融合、验证分页 KV 与真实派发 | [算子接入](../rsinfer-ascend-kernel/SKILL.md) |
| 定位首个错误 token、ACLNN / AICore 错误、ACL Graph 回放、HBM OOM 或 HCCL 卡住 | [推理诊断](../rsinfer-ascend-debug/SKILL.md) |

已有 AKA task ID、待接回的候选，或要将已定位的算子交给 AKA 研究时，使用 [AKA 工作流](../optimize-kernels-with-aka/SKILL.md)。部署命令、复验步骤与历次实测按任务查 [Ascend 文档导航](../../../doc/README.md)。

## 目录地图

| 要找的代码 | 位置与用途 |
| --- | --- |
| 部署和测量 | [deploy](../../../deploy/README.md) 下的 `rsinfer.py` 管原生引擎，`models/` 按 checkpoint 选配方，`bench_ascend.py` 跑 Qwen 流式请求或 GLM AISBench，`memcache.py` 管独立缓存服务。 |
| 设备与框架 | [framework.toml](../../../engine/framework.toml) 列出变更后需要整引擎重启的模块，包括 `platforms.ascend`。控制面、变更事务、进程与单元重载分别在 `engine/core/{control,change,supervisor,nodeagent}` 和 `engine/unit/runtime`。 |
| 请求到执行 | HTTP、分词和回包在 `engine/frontend/`。调度与状态分配在 `engine/core/{sched,state/alloc}`，请求与策略在 `engine/{sched/reqs,policies}`。worker 组批和通信在 `engine/worker/{batch,comm}`，执行与录图在 `engine/executors/pytorch` 和 `engine/core/plan`。 |
| 模型与算子 | `engine/models/{qwen3,glm53}` 实现原生 Ascend 模型。沿 `engine/layers/` 找调用点，`engine/core/kernels` 选方案，`engine/kernels/{ascend,reference}` 放实现与参考。 |
| 权重与状态 | `engine/core/state/{weights,holder}` 管权重和设备内存预算，`engine/state/{kv_paged,glm53_cache,gdn_slots}` 管分页、混合前缀和递归状态。状态所有者及其动作见 [状态变更协议](../../../doc/specs/state-changes.md)。 |
| 验证与故障记录 | `tests/units/` 对应单元，`tests/framework/` 测框架，`tests/framework/formal/` 验证协议，`tests/scenarios/` 跑整引擎场景。验证命令见 [AGENTS.md](../../../AGENTS.md)，设计按主题查 [规格索引](../../../doc/specs/README.md)，热改机制异常按 [故障报告格式](../../../doc/framework-bugs/README.md) 记录。 |

引擎从快照根目录导入 `core.*`、`models.*` 等包。改一个单元前先看它的 `unit.toml` 和服务使用方，跨单元、跨进程的数据定义在 `engine/core/contracts/`。重载对依赖方的影响见 [单元运行时](../../../doc/specs/unit-runtime.md)。

## 在跑着的引擎上热改

### 起一个引擎

从仓库根目录用目标 worker 的 Python。设好 `RSI_SOURCE`（含 `engine/` 的工作区根目录）、`RSI_RUN`（本实例的运行目录）和固定测试输入 `RSI_PROMPT`。已有实例先查状态。新建实例时按模型选择 die、端口、精度和并行参数，保留实际启动命令，`serve --dry-run` 可在单独的运行目录预览配置与拓扑。 [原生和对照服务的启动示例](../../../deploy/ascend.md)

```sh
rsi() {
    python deploy/rsinfer.py "$@"
}
rsi status --run "$RSI_RUN" --json
```

### 改一次

动手前记录已提交 identity、有效配置和基线请求结果，留好能恢复该版本的源码。在 `status --json` 的 `control_status.procs` 中按 worker 记录 `local_pid`，顶层 `pid` 对应控制面。

```sh
rsi control --run "$RSI_RUN" check --source "$RSI_SOURCE"
rsi control --run "$RSI_RUN" apply --source "$RSI_SOURCE" --dry-run
rsi control --run "$RSI_RUN" apply --source "$RSI_SOURCE" --wait-empty 120
rsi status --run "$RSI_RUN" --json
rsi complete "$RSI_PROMPT" --run "$RSI_RUN" --max-tokens 16
```

检查清单和服务依赖后，读 dry-run 的动作与诊断。看到 `escalate.restart_engine` 时，按原配方重启目标实例。允许热更时核对 `committed`、新 identity、各 rank 的实际动作和请求输出。只改配置可给 `apply` 传 `--config`。仅修改 live 配置键可带流量生效。代码、状态等变更需要空引擎。`--wait-empty` 期间新请求收到 503，在途请求可跑完，超时会重新开门。 [排空语义](../../../doc/specs/empty-apply.md)

### 在引擎里做检查（排查和验证）

1. 从与目标 identity 对应的源码创建独立探针工作区，在要观察的单元中标记临时代码 `# PROBE`。
   ```sh
   RSI_PROBE=$(mktemp -d)
   cp -a "$RSI_SOURCE/engine" "$RSI_PROBE/engine"
   ```
2. 按上节流程把 `--source` 指向 `"$RSI_PROBE"`，重放固定请求。读 `<run>/host-a/snapshots/.run/<角色实例>/log` 的新增部分，关联 identity、rank、阶段与形状。`control logs` 查变更事务，`control introspect` 查各进程当前的单元和读取记录。ACL Graph 的 Python 探针放在图外。 [NPU 采集方法](../../../doc/2026-11-26-ascend-npu-workflow.md#2-用实际-worker-采集性能)
3. 结束采集后，重新对已确认的无探针源码执行 check、dry-run、apply，恢复本轮改动过的配置，再发同一请求。用干净的 worker 路径做正式计时，把需要长期防止的回归留在对应测试层。

### 撤回没成功、引擎要求恢复的时候

- `aborted` 后用 `control introspect` 和原请求确认旧版已恢复。撤销已提交的候选时，把留存的源码和变更前配置作为一次新变更提交并验证。 [控制面与事务](../../../doc/specs/control-plane.md)
- 出现 `change.abort_unacknowledged` 或 `change.recovery_required` 时，`status` 可能仍显示上次确认的单元状态。保存事务诊断和各 rank 日志，用 `introspect` 查询现场。从已确认版本和配置对该实例执行 `stop --run` 和原 `serve` 命令，检查新 PID、identity 和原请求。

## 状态、权重与并行变更

| 任务 | 验证重点 |
| --- | --- |
| 释放设备内存 | `sleep --run "$RSI_RUN" --wait-empty 120` 排空、释放权重并缩小状态池。调用 `wake --run "$RSI_RUN"` 后，从当前 checkpoint 重新加载，核对服务与输出。 |
| 切换权重 | 给 `control apply` 传 `weights.path` 配置并排空，先确认 checkpoint 与当前模型结构兼容。内部状态动作 `refit` 会复用原张量地址。GLM 还要核对录图使用的派生权重内容。 [GLM refit 回归](../../../doc/framework-bugs/2026-11-26-glm53-packed-weights-refit.md) |
| 投机解码 | 原生 GLM 用带 MTP 层的完整 checkpoint。记录草稿数、接受数和校验耗时，和同配置关闭投机的结果对照。 [MTP 协议](../../../doc/specs/mtp.md) · [GLM 原生配方](../../../deploy/ascend.md#rsinfer-原生-glm-53-flash) |
| 改通信拓扑 | 改变原生 GLM 的 die 列表或调度域数量时，用新配方参数重启引擎，核对各域的 TP/EP 成员。重载通信单元时，确认所有 rank 按同一顺序重建 HCCL 子组。 [GLM 拓扑配方](../../../deploy/models/glm53_native.py) · [HCCL 组重建记录](../../../doc/framework-bugs/2026-11-26-hccl-subgroup-rejoin.md) |
