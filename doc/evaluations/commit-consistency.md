# Commit 丢失的一致性修复：A/B/C 对抗验证

本记录只处理 PR #1 的第一个 P1 问题：Commit 未送达时，协调者仍报告提交成功；未收到 Commit 的参与者超时回滚，造成版本分裂。基线为 `framework-base` 的 `b006af178dc77c75faba0316fc7d1fca239ca9e2`。其他四个 review 问题不在本次修改范围内。

## 分支与实验结果

每个分支有自己的 worktree，位于仓库旁的 `agxinfer-worktrees/`。A 保留原始生产代码；B 是候选修复；C 是故意违反修复目标的实验代码，不应合并到生产分支。

| 角色 | 分支（共同前缀 `gw/framework-base/`） | 初始测试提交 | 初始 3 例 | 加强测试提交 | 最终 12 例 |
| --- | --- | --- | --- | --- | --- |
| A：原始实现 | `commit-consistency-baseline` | `99d8bdc` | 3 failed | `fac0090` | 12 failed |
| B：修复实现 | `commit-consistency` | `25cf3a9` | 3 passed | `5c9939c` | 12 passed |
| C：对抗实现 | `commit-consistency-adversarial` | `a2fd1c8` | 3 passed | `7fc2e5e` | 4 failed, 8 passed |

这些都是实际运行结果，没有将预期失败标为 `xfail`，也没有跳过失败用例。最终 A、B、C 使用逐字相同的 `tests/change/test_commit_consistency.py`，SHA-256 为：

```text
ceffd2fa05505a410e5e4c19cee0613bc24d0fee22bbbdbd15c2a362cda9caaf
```

## 修复契约与实现

1. 协调者必须确认所有参与者的提交决定后才能发 Resume。成功回执必须匹配消息类型、change ID 和候选 identity；回滚回执不能充当提交成功。
2. Commit 与 Resume 请求或回执丢失时，重试未确认的参与者。重复提交不能重复执行 commit hook。
3. 一旦 Execute 完成，参与者可能已经处于全局提交决定中。此时超时只能保持暂停，不能单方面回滚。执行前的超时回滚行为保持原有语义。
4. 永久无法确认时，返回 `failed` 和 `change.recovery_required`，禁止后续变更，不发 Abort。这里的 `failed` 表示提交完成情况无法确认，不表示已经回滚；返回的 `new_identity` 是已决定提交的候选身份。
5. 新增和重启的候选进程也必须从暂停状态启动，经过相同的 Resume 屏障才能服务。

B 在发出 Commit 前持久化 `commit_decided` 事件；参与者收到 Commit 后切换身份并保持暂停，回复 Committed。全部确认后，协调者更新 Supervisor 的进程规格并发送 Resume，全部确认恢复后才返回 `committed`。默认最多尝试 3 次，重试间隔 0.1 秒，分别由 `change.commit_attempts` 和 `change.commit_retry_delay_s` 配置。

Commit 的语义发生变化，schema 从 `change.commit@1` 升为 `change.commit@2`，新增 `change.committed@1` 和 `change.resume@1`。这是框架核心协议修改，需要整体更新框架进程，不能将旧协调者和新参与者混用。

本修复优先保证一致性：永久丢包会损失可用性。Commit 无法全部确认时参与者保持暂停；Resume 无法全部确认时，部分进程可能已经服务，但它们都已提交同一个身份。恢复需要按现有框架策略重启整个引擎；本次没有实现崩溃后的日志重放或自动在线恢复。

## C 如何逃过第一轮测试

C 只在 B 的参与者实现中添加两行生产逻辑：

```python
# on_commit: 本地提交后立即服务，绕过全局确认屏障。
self.host.paused = False

# tick: 发现执行／提交阶段超时后重新暂停。
self.host.paused = True
```

它没有修改测试、检查测试名称或伪造测试结果。其错误假设是：本地提交就可以先恢复，随后用超时暂停补救。

初始三个用例检查 Commit 请求丢失、Commit 回执丢失和永久丢失后的最终状态。C 在超时时重新暂停，因而能通过全部初始用例，但“收到 Commit 到发现超时”之间已经可能提前服务。

加强测试在每次消息交付之后记录实际参与者的身份、暂停状态和单元行为，检查是否存在“其他参与者仍是旧身份，本参与者已经恢复服务”的时刻。分别丢弃第一个／第二个参与者的请求或回执，形成四种调度。C 的四种情况全部失败，错误示例为：

```text
AssertionError: served before all participants committed:
[(ProcId(host='host-0', role='worker', index=0), 2)]
```

最终 12 例还覆盖 Resume 请求／回执丢失的幂等重试、Resume 永久丢失后拒绝后续变更，以及新增／重启进程在全局提交前保持暂停。

## 测试边界

测试使用真实 Coordinator、Participant、Supervisor 和 ProcessHost，以及真实 UnitRuntime 加载的单元。单元独立记录行为版本与 commit 次数；断言同时检查协调者返回值、各参与者身份、行为版本、暂停状态和 Supervisor 记录，避免只验证协调者自己声称成功。

故障注入替换 RPC 传输边界、物化和进程启动；使用两个独立 Host/Runtime 对象，并推进参与者 deadline 触发真实超时处理。新增／重启用例仍经过真实 Supervisor.launch/relaunch 和 ProcessHost 初始化，但不创建 OS 子进程。本组测试没有验证真实网络的全部交错、节点代理物化或真实 GPU。这是一次实际的 mutation 逃逸与测试加强实验，不是独立多 agent 审核，也不证明不存在其他对抗实现。

环境为 macOS arm64、CPython 3.12.13，依赖使用 `uv.lock`。B 的 `pyright` 结果为 0 errors；三份最终测试文件内容相同，`git diff --check` 通过。

同时运行了既有全量测试。原始实现：71 passed / 24 failed。B 在初始三例加入后的全量结果：74 passed / 24 failed，失败的既有测试名称与基线一致。其中 22 例因 macOS 上只读快照目录 rename 的 PermissionError 失败，另有 TCP 背压测试失败，以及 Python 3.12 下测试遍历 `sys.modules` 时访问 `typing.io.__spec__` 的 AttributeError。本次未修改这些问题；全量检查没有通过，现有多进程事务测试也因此尚未验证到提交阶段。

## 复现

在任意一个对应 worktree 中执行：

```sh
uv run --frozen --python 3.12 pytest tests/change/test_commit_consistency.py -q
```

预期 A/C 的退出码为 1，B 的退出码为 0。B 的静态检查：

```sh
uv run --frozen --python 3.12 pyright
```

初始实验也已保留提交，可以用独立临时 worktree 复现，例如 C 在加强测试前的逃逸：

```sh
git worktree add --detach /tmp/agxinfer-commit-C-initial a2fd1c8
cd /tmp/agxinfer-commit-C-initial
uv run --frozen --python 3.12 pytest tests/change/test_commit_consistency.py -q
# 3 passed
```

检查 C 相对 B 的生产代码变化、以及 A 未修改生产代码：

```sh
git diff 5c9939c 7fc2e5e -- engine
git diff b006af1 fac0090 -- engine
# 第二条应无输出。
```
