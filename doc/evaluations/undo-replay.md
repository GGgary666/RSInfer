# 防止已撤销事务被迟到消息重新执行

基线为 `00f5426`：生产代码等于 `framework-base@010c075`，包含 PR #9 的初始反例。此分支只修复本项问题；其他三项由独立分支处理。

## 修复与范围

Prepare 和 Pause 在 undone 阶段返回失败，保留暂停状态，等待协调者 Abort；不重新执行动作，也不提前恢复服务。

仅修改一个生产文件：`engine/core/change/participant.py`。新增 `tests/adversarial/test_undo_replay.py` 和本文。fix、加强测试、文档放在同一个提交中。

## A/B/C 实测

- A：`gw/framework-base/adversarial-fixes-baseline`，生产代码未修复。
- B：`gw/framework-base/undo-replay`，正确修复。
- C：`gw/framework-base/undo-replay-adversarial`，故意错误的实现，仅作实验。

C 的错误：在拒绝迟到 Pause 之前调用 _resume，提前恢复旧版本服务。最终状态仍可满足原测试，所以初始 2 例通过。加强测试在 Abort 之前检查暂停状态，并确认后续新事务可正常提交。

| 版本 | 对应初始测试 | 初始 + 加强测试 |
| --- | --- | --- |
| A | 至少一例失败 | 0 通过 / 5 失败 |
| B | 2 通过 | 5 通过 / 0 失败 |
| C | 2 通过 | 3 通过 / 2 失败 |

A/B/C 使用相同的初始及加强测试；没有 xfail、skip 或放宽原断言。表中只运行本项对应的测试，不把其他尚未应用的修复算入本项结果。

## 复现

在对应 worktree 的项目环境中执行初始测试：

```sh
python -m pytest tests/adversarial/test_transaction_boundaries.py -q -k 'late_pause or pause-after-failed-execute'
```

执行初始及加强测试：

```sh
python -m pytest tests/adversarial/test_transaction_boundaries.py tests/adversarial/test_undo_replay.py -q -k 'late_pause or pause-after-failed-execute or undone_change'
```

四项修复在 `gw/framework-base/adversarial-fixes` 集成验证，完整反例集命令为 `python -m pytest tests/adversarial -q`。

本次依用户授权在 macOS / CPython 3.12.13 本地验证；没有添加非 Linux 兼容代码。Linux 验收仍需 CI。协议用例使用真实 Participant/UnitRuntime，并保留 PR #9 中的真实 RPC 用例；它们不依赖快照目录 rename 的平台适配。既有平台失败不作为本修复的产品缺陷。
