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

### 已保留的初始版本

| 版本 | 分支 | 初始提交 | 初始共同用例 |
|---|---|---|---|
| A | `gw/dev/cuda-graph-abc-a` | `d884dda` | 1 failed：没有就绪 decode graph |
| B | `gw/dev/cuda-graph` | `bd35f4f` | 1 passed |
| C eager | `gw/dev/cuda-graph-abc-c-eager` | `1ebebc9` | 1 passed，逃逸 |
| C catalog | `gw/dev/cuda-graph-abc-c-catalog` | `26b2080` | 1 passed，逃逸 |

两个 C 还各自通过了修改前的 `tests/gpu/test_cuda_graph.py` 全部 6 项；连同共同初始用例，分别为 **7 passed / 6.51 秒**和 **7 passed / 10.77 秒**。本轮未在 C 上重跑 HTTP apply 文件或吞吐 benchmark。

初始共同测试 SHA256：`3ab6fbb6780006cd08e63839a86f2e54b07c33c9b566ed9146c36f830e18e6d1`。

### 测试加强

共同文件为 `tests/gpu/test_cuda_graph_adversarial.py`，最终 SHA256：`4c5094a0b9e5d6e8feb49ddb97d9ba8106aed0c251a27947d7441a5e30d4608a`。四份文件逐字相同。

- 保留原有 eager/graph logits 与 token 对照。
- 在构建完成后观察底层 `torch.cuda.CUDAGraph.replay`，确认 prefill 不回放、decode 依次选择 bucket 1/2/4 并各回放一次，包含三行落入四行桶的 padding。观察器调用原始 replay，不替代 GPU 执行。
- 在同一 catalog 实例新增一个会改变 linear 结果的 solution，确认目录版本变化、batch generation 不变且新 solution 已可选。提交 decode 必须抛出 StaleGraph，且没有任何 CUDA replay，所有 batch/KV/GDN buffer 逐位不变。

A 通过相同的就绪 graph 断言拒绝，不把旧版本不存在的新模块/方法造成的导入错误当证据。最终共同用例对平台集合做显式 `set[str]` 类型声明，也消除了 A 上初始测试的静态类型提示；没有改变验收逻辑。

## 最终 A/B/C 实测

| 版本 | 最终测试提交 | 同一套 3 项测试 | 被拒绝的行为 |
|---|---|---|---|
| A | `5cf936b` | 3 failed，4.32 秒 | 没有就绪 decode graph |
| B | `fc94200` | 3 passed（包含在最终 9 passed 集合中） | 正确实现通过 |
| C eager | `d3b137d` | 2 passed / 1 failed，5.07 秒 | decode 实际 replay 次数为 0 |
| C catalog | `70cc0eb` | 2 passed / 1 failed，8.44 秒 | catalog 变化后没有抛出 StaleGraph |

A/C 均为正常 pytest 断言失败、退出码 1；B 的最终 GPU 与全量组合命令退出码 0，包含 `RSINFER_GPU_TEST_OK` 和 `ABC_B_FINAL_OK`。两个 C 各由一个加强用例检出，不声称每一项测试都能拒绝它们。

A 的 engine 与 `07442d6` 逐字相同。相对 B，C eager 只多保存一个 forward 方法并将 replay 改为 eager forward（+2/-1）；C catalog 只删除版本检查两行。加强前后 C 的生产改动没有变化，只增强了测试。

```sh
git diff 07442d6 5cf936b -- engine   # 无输出
git diff fc94200 d3b137d -- engine  # eager 替代 replay
git diff fc94200 70cc0eb -- engine  # 删除 catalog version 检查
```

## B 的回归与验证范围

- 代码简化与 master 合并：`bd35f4f`；加强测试：`fc94200`。两者的 engine 内容相同。
- 本地 pyright：0 errors；四个最终版本的共同测试均通过静态检查。
- 本地 executor/kernel/state/deps 定向回归：95 passed。
- 首轮真实 GPU 回归（原有 6 项、HTTP apply/配置切换 3 项、初始验收 1 项）：10 passed / 92.64 秒。
- 最终 GPU 回归（原有 6 项、加强后的共同验收 3 项）：9 passed / 7.17 秒。HTTP apply 文件在这两轮之间没有改变，没有重复跑它。
- 最终 Linux 全量 `pytest -n 8 -q`：570 passed / 75.13 秒。

第一次 Linux 全量为 569 passed / 1 failed。调度器多进程测试裁剪工作区时保留了 KV 单元，却遗漏声明 `plan.cuda_graph` 的 `core.plan`，导致 `manifest.undeclared_read`；补齐测试工作区的配置声明来源后，单独复验 1 passed / 4.97 秒，最终全量通过。没有改生产配置校验来绕过此失败。

远程机器为 NVIDIA SM90-80GB/SM90，Python 3.12.7、PyTorch 2.11.0+cu130。远程 pyright 因缺 Node 且无法访问下载站未运行；静态检查采用本地已有环境。首次 C eager 任务在安装 release 时遇到 runner staging 竞争，尚未启动 pytest；确认失败后以同一 RID 重提，得到上述 7 passed，不将执行器异常算作测试检出。

本轮不重跑吞吐 benchmark、完整 27B 真权重或 SM100。既有 formal 回归不能替代本轮 CUDA Graph 对抗实验；这两种 C 被检出也不证明不存在其它错误实现。

## OSS 执行记录

所有任务固定 release；同一 GPU 环境复用，不重新安装依赖。日志保存在本地 `.ossrun-artifacts/abc-*.log`，原始 terminal.log 也保留在对应 OSS command 目录。

| 运行 | Release | Command |
|---|---|---|
| A 初始 | `20261110T112113Z_07442d6-dirty_95685e43` | `cmd_20261110T112210Z_abc-a-initial_efc23046` |
| B 初始 GPU 与全量 | `20261110T111932Z_3dc7b1a-dirty_ff11cba5` | `cmd_20261110T112200Z_abc-b-gpu_945fe2bd` |
| 调度器夹具修正复验 | `20261110T112825Z_3dc7b1a-dirty_96d651b9` | `cmd_20261110T112840Z_abc-scheduler-fix_d3dc797d` |
| C eager 初始，执行器失败后重提 | `20261110T112945Z_bd35f4f-dirty_ff9fb36d` | `cmd_20261110T113849Z_abc-c-eager-retry_0fc88f0d` |
| C catalog 初始 | `20261110T112949Z_bd35f4f-dirty_84863b25` | `cmd_20261110T113028Z_abc-c-catalog-initial_67cbab38` |
| B 最终 GPU 与全量 | `20261110T114241Z_bd35f4f-dirty_721613f1` | `cmd_20261110T114324Z_abc-b-final_9f2f0474` |
| A 最终 | `20261110T114245Z_d884dda-dirty_ba55ccf0` | `cmd_20261110T114343Z_abc-a-final_d7ba13d6` |
| C eager 最终 | `20261110T114248Z_1ebebc9-dirty_d55ebb26` | `cmd_20261110T114345Z_abc-c-eager-final_0fef28b8` |
| C catalog 最终 | `20261110T114252Z_26b2080-dirty_51e472a1` | `cmd_20261110T114347Z_abc-c-catalog-final_95a5c29d` |

在各版本 worktree 的 GPU 环境运行同一条命令：

```sh
uv run --no-sync pytest tests/gpu/test_cuda_graph_adversarial.py
```

通过 OSS 复现最终版本时，使用表中 RID：

```sh
bash oss_run_rsinfer_cuda_graph/scripts/ossrun/run.sh --release <RID> \
  --inline 'bash oss_run_rsinfer_cuda_graph/scripts/ossrun/gpu.sh test tests/gpu/test_cuda_graph_adversarial.py' \
  --name graph-abc-reproduce
```

A/C 的预期退出码为 1，必须核对具体 pytest 断言；`agent exception`、导入失败或环境失败不算成功检出。B 预期退出码为 0，并包含 `RSINFER_GPU_TEST_OK`。

初始逃逸可在 `1ebebc9`、`26b2080` 的临时 worktree 复现，共同文件当时只有 1 个用例；加上 `tests/gpu/test_cuda_graph.py` 就是上述 C 各 7 passed 的集合。最终 A/C 分支保留测试加强提交，B 的生产实现不含任何 C 改动。
