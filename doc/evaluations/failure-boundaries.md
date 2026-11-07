# 停止、回滚、内部队列和进程环境的 A/B/C 验证

基线：`framework-base@8d30a47`（已合并 PR #9）。本次修复四个独立复现的问题，生产修改限于 coordinator、participant、supervisor、queue 和 nodeagent 五个文件。

## 行为约束与修复

1. **停止需要确认。** LOST 主机上的进程可能仍在服务，`stop()` 返回 `supervisor.stop_unconfirmed`。协调者不忘记该进程，不把它放进已停止集合，返回 `failed/change.recovery_required` 并拒绝后续变更。重启和恢复旧版本也必须先确认停止，再 spawn；停止失败时保留进程和恢复信息。
2. **回滚失败不能恢复服务。** 动作 abort、反向 reload 或清理异常被保存在本次变更中；参与者保持暂停，重复 Abort 返回原失败，新 Prepare、Execute、Commit 和 Resume 不能绕过它。同步 prepare、分片 prepare、租约到期及 execute 的撤销共用这一约束。协调者传播诊断，只把匹配 change_id 且 rolled_back 的回执视作确认；缺失或无效回执以及恢复失败都冻结后续变更。正常成功回滚仍返回 aborted。
3. **优先级不豁免过期检查。** fault 队列中的 internal 变更同样检查 computed_under；丢弃过期项不重置普通 internal 队列的公平性计数。operator 变更不因该规则丢弃。
4. **进程环境不能重写运行约束。** 先合并继承环境与 ProcSpec.env，再清理代码来源变量，最后设置快照路径、只读快照的 bytecode 设置和 GPU 分配。合法自定义变量仍可覆盖继承值。

没有增加协议消息或在线恢复机制；进入 recovery_required 后仍按现有机制要求恢复/重启引擎。无法联系的旧进程不能靠本次修复保证停止，只能诚实报告未确认并冻结变更。

## 分支与测试

| 角色 | 本地分支 | 初始 16 项 | 加强后共 30 项 |
| --- | --- | --- | --- |
| A：原生产代码 | `gw/framework-base/failure-boundaries-baseline` | 5 通过 / 11 失败 | 6 通过 / 24 失败 |
| B：修复 | `gw/framework-base/failure-boundaries` | 16 通过 | 30 通过 |
| C：故意错误的实现 | `gw/framework-base/failure-boundaries-adversarial` | 16 通过 | 22 通过 / 8 失败 |

C 从 B 构造四种错误：停止未确认时仅报告 aborted 而不冻结；回滚失败虽然返回 Failed 却解除暂停并清空事务；丢弃过期 fault 时重置公平性计数；环境清理误删合法自定义变量。这些错误全部通过初始测试，分别被加强后的行为断言捕获。A、C 仅保存在本地，PR 只提交 B。

初始测试：`tests/framework/adversarial/test_failure_boundaries.py`。加强测试：`tests/framework/adversarial/test_failure_boundary_adversaries.py`。同一份测试在 A/B/C 上运行，不包含 xfail 或按实现分支跳过的逻辑。测试使用真实协调者、参与者和运行时，在通信或 hook 边界注入故障；不等价于 Linux 多进程网络分区实验。

```sh
uv run pytest tests/framework/adversarial/test_failure_boundaries.py tests/framework/adversarial/test_failure_boundary_adversaries.py -q
uv run pyright
uv run pytest
```

已有 `test_in_doubt.py` 的丢失 Abort 用例同步调整：决定仍为 abort，参与者仍可查询决定完成回滚，但协调者尚未收到确认时应报告 failed 并冻结，不能提前声称恢复完成。

## 本地回归结果

macOS / Python 3.12：pyright 为 0 errors。未修复基线的既有集合为 141 通过、53 失败；B 完整集合为 171 通过、53 失败，失败用例集合完全相同。53 项分别是 51 项快照物化 PermissionError、1 项 TCP 背压边界、1 项 typing.io 平台差异。未增加 macOS 兼容代码或 CI。

Linux 全量验证由仓库现有 ubuntu-latest CI 执行；本地结果不能代替 Linux 验证。
