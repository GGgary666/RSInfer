---
name: optimize-kernels-with-aka
description: 已定位 RSInfer 算子且需要 AKA 搜索、有现成 task ID 或待接回的候选时，建立或接手研究任务。Ascend 使用目标主机的本地 command 契约，按真实服务调用验收。
---

# 用 AKA 优化 RSInfer 算子

从真实 worker 取当前实现、调用次数、代表性 shape/stride、精度、状态和布局。用它作 baseline，将候选起点放入 seed。参考、输入生成器、计时与判定放入独立的 `contract/`。多 rank 算子的验证器还需复现各 rank 的输入和通信组。以同一输入复测 baseline，确认抽取后的输出与开销可代表服务调用。 [Ascend 算子接入](../rsinfer-ascend-kernel/SKILL.md)

## 在 Ascend 主机建立契约

在选定的研究 Python 环境使用仓库固定的 SDK。缺少时从 RSInfer 根目录执行 `python -m pip install -r .claude/skills/optimize-kernels-with-aka/requirements.txt`。需要查 SDK 源码时运行 `git submodule update --init -- third_party/aka`。升级时同步更新依赖、子模块指针和任务接口链接。 [SDK 版本](requirements.txt)

- 选 `execution={"backend": "local"}`。在执行主机加载该设备的 CANN 环境，验证器内核对 `torch_npu`、OPP、实际可见 die 和后端。对 Ascend 异步调用先预热，在计时区间两端同步设备，分别记录算子耗时和含适配的调用耗时。
- 在 `contract/contract.json` 中设 `"adapter": "command"`。`config.command` 为 argv，例如 `["python", "{contract}/verify.py", "{candidate}"]`，其中 `python` 应是上述环境的解释器。`config.instructions` 写明输入输出、状态更新与误差，`config.delivery` 写候选入口。验证器加载传入目录中的实现，在冻结的 case 上核对结果和时延。 [固定版本的任务接口](https://github.com/alibaba/atrex-kernel-agent/blob/e6e2926d5251c7b815597af165d6396c42e4001c/docs/task-execution.md)
- 验证器向 stdout 写一行 `AKA_TASK_RESULT=` 加 JSON。给出 `verdict`、`complete` 与 `metrics`，每项指标的 `name`、`value`、`unit` 与提交目标对齐。保留逐 case 结果、设备和软件版本。

## 提交或接手

在含 `contract/`、`baseline/`、`seed/` 的实验目录执行。在所选研究 agent 已配置时可异步启动任务。下例用 Codex，`latency_us` 需与验证器输出一致。

```python
import json
from pathlib import Path

import aka

root = Path("aka-tasks").resolve()
job = aka.submit(
    contract="contract", baseline="baseline", seed="seed",
    execution={"backend": "local"}, agent={"runtime": "codex"},
    objective={"metrics": [{"name": "latency_us", "unit": "us", "direction": "minimize"}]},
    key="prefill-projection-v1", root=root,
)
Path("aka-job.json").write_text(json.dumps({"root": str(root), "task_id": job.id}))
```

已有任务用记录中的 `task_id` 和 `root` 调用 `aka.job(task_id, root=root)` 重连。通过 `job.events()` 和 `job.results()` 查看进展。从 `results["candidates"]` 选已验收的 ID，用 `job.export(candidate_id, destination)` 导出。重复提交相同任务可复用 key，形状、基线或验证规则变化时另建 key。

## 回接与验收

已有候选可从此处开始。在原 worker 通过热更验证宿主和扩展能同进程加载，检查目标阶段实际命中。用同样的激活、分页状态与 stride 对比基线，计入重排、拷贝、分配、同步和通信。若回接需要改绑定或实现，重新执行该契约。 [引擎内检查](../rsinfer/SKILL.md#在引擎里做检查排查和验证)

分别记录裸算子、含适配的调用和用户负载结果。按实际收益选择形状和阶段，保留回退路径。用 `job.feedback(candidate_id, value)` 回传采用范围或回退依据，同时记录 RSInfer revision、契约 ID、候选 ID 和测量产物。 [服务对照与验收](../rsinfer-ascend-optimize/SKILL.md#验收并记录)
