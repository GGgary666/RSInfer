# 按文件及 artifact 内容更新清理标记

基线为 `00f5426`：生产代码等于 `framework-base@010c075`，包含 PR #9 的初始反例。此分支只修复本项问题；其他三项由独立分支处理。

## 修复与范围

清理后检查 change-set 元数据是否存在，并检查其引用的非删除文件与 artifact blob；任一缺失即标记 content_purged。保留原有宽限期及保留规则，删除文件的 None 值不作为 blob hash。

仅修改一个生产文件：`engine/core/change/eventlog.py`。新增 `tests/adversarial/test_retention_content_flags.py` 和本文。fix、加强测试、文档放在同一个提交中。

## A/B/C 实测

- A：`gw/framework-base/adversarial-fixes-baseline`，生产代码未修复。
- B：`gw/framework-base/retention-content-flags`，正确修复。
- C：`gw/framework-base/retention-content-flags-adversarial`，故意错误的实现，仅作实验。

C 的错误：只检查 files，遗漏 artifacts。初始 2 例全部通过；加强测试组合 file/artifact 与删除、宽限期、显式保留，artifact 被删除时将 C 检出。

| 版本 | 对应初始测试 | 初始 + 加强测试 |
| --- | --- | --- |
| A | 至少一例失败 | 5 通过 / 3 失败 |
| B | 2 通过 | 8 通过 / 0 失败 |
| C | 2 通过 | 7 通过 / 1 失败 |

A/B/C 使用相同的初始及加强测试；没有 xfail、skip 或放宽原断言。表中只运行本项对应的测试，不把其他尚未应用的修复算入本项结果。

## 复现

在对应 worktree 的项目环境中执行初始测试：

```sh
python -m pytest tests/adversarial/test_retention_evidence.py -q -k 'aborted_change'
```

执行初始及加强测试：

```sh
python -m pytest tests/adversarial/test_retention_evidence.py tests/adversarial/test_retention_content_flags.py -q -k 'aborted_change or purge_flags'
```

四项修复在 `gw/framework-base/adversarial-fixes` 集成验证，完整反例集命令为 `python -m pytest tests/adversarial -q`。

本次依用户授权在 macOS / CPython 3.12.13 本地验证；没有添加非 Linux 兼容代码。Linux 验收仍需 CI。协议用例使用真实 Participant/UnitRuntime，并保留 PR #9 中的真实 RPC 用例；它们不依赖快照目录 rename 的平台适配。既有平台失败不作为本修复的产品缺陷。

## 四项修复的集成结果

集成分支 `gw/framework-base/adversarial-fixes` 包含四个独立 fix/test/docs 提交。生产代码合计只改三个文件，17 行新增、4 行删除；非法的 Prepare/Pause 在入口返回失败，staging 声明失败恢复已有配置且不切换代码路径。

- 原 PR #9 的 14 例加本次加强的 14 例：**28 例全部通过**。
- 四个 C 分支均能通过各自初始用例；加强后分别有 2、2、2、1 例失败。A/B/C 使用相同测试文件。
- 全仓库 Pyright 为 0 errors、0 warnings；`git diff --check` 通过。
- 第一次全量本地运行：140 通过、54 失败。除基线已有的 53 项平台相关失败外，SHM 顺序测试出现一次 `PeerLost`；该用例在 A 和集成 B 上分别单独重跑都通过。
- 为检查这次额外失败，重新运行基线原先通过的 113 例，加全部 28 例对抗测试，结果 **141 通过**。这组运行未包含基线已有的 53 个失败用例，不等同于全量测试通过。

未修改 IPC 实现，也未扩大到非 Linux 兼容修复。上述全部是本机结果；Linux CI 尚未验证修复分支。
