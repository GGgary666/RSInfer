# 配置声明失败时恢复 staging 前的环境

基线为 `00f5426`：生产代码等于 `framework-base@010c075`，包含 PR #9 的初始反例。此分支只修复本项问题；其他三项由独立分支处理。

## 修复与范围

在声明候选配置前保存原配置；声明失败则完整恢复。只有声明成功后才修改代码搜索路径和 staged root，避免半完成的声明及目录残留。

仅修改一个生产文件：`engine/unit/runtime/runtime.py`。新增 `tests/framework/adversarial/test_staging_atomicity.py` 和本文。fix、加强测试、文档放在同一个提交中。

## A/B/C 实测

- A：`gw/framework-base/adversarial-fixes-baseline`，生产代码未修复。
- B：`gw/framework-base/staging-atomicity`，正确修复。
- C：`gw/framework-base/staging-atomicity-adversarial`，故意错误的实现，仅作实验。

C 的错误：声明失败时清空整个配置表。原测试开始时没有既有配置，因此初始 4 例全部通过；加强测试先设定已有配置为 42，再制造类型错误或所有权冲突，检查原值、代码搜索路径，以及下一合法候选。

| 版本 | 对应初始测试 | 初始 + 加强测试 |
| --- | --- | --- |
| A | 至少一例失败 | 2 通过 / 4 失败 |
| B | 4 通过 | 6 通过 / 0 失败 |
| C | 4 通过 | 4 通过 / 2 失败 |

A/B/C 使用相同的初始及加强测试；没有 xfail、skip 或放宽原断言。表中只运行本项对应的测试，不把其他尚未应用的修复算入本项结果。

## 复现

在对应 worktree 的项目环境中执行初始测试：

```sh
python -m pytest tests/framework/adversarial/test_transaction_boundaries.py -q -k 'rejected_staging'
```

执行初始及加强测试：

```sh
python -m pytest tests/framework/adversarial/test_transaction_boundaries.py tests/framework/adversarial/test_staging_atomicity.py -q -k 'rejected_staging or failed_staging'
```

四项修复在 `gw/framework-base/adversarial-fixes` 集成验证，完整反例集命令为 `python -m pytest tests/framework/adversarial -q`。

本次依用户授权在 macOS / CPython 3.12.13 本地验证；没有添加非 Linux 兼容代码。Linux 验收仍需 CI。协议用例使用真实 Participant/UnitRuntime，并保留 PR #9 中的真实 RPC 用例；它们不依赖快照目录 rename 的平台适配。既有平台失败不作为本修复的产品缺陷。
