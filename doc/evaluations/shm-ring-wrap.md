# SHM ring 回绕阻塞：A/B/C 验证

基于最新拉取的 `framework-base@fface7edb572241942e94f88cd5209d329f1a0e3`，其中已合入第一个问题的 PR #6。本次只处理第二个问题：变长消息跨 ring 尾部时，writer 可能永远等不到可用空间。

## 修复范围与原因

生产代码只修改 `engine/core/ipc/stream.py` 的 `ShmWriter.write()`，新增 7 行。另增加一个测试文件和本文档，不修改其他生产文件、HTML 或既有 Markdown。

64 字节 ring 的可复现例子：

1. 写入 20 字节 payload，加长度头与对齐后占 24 字节。
2. reader 消费这条消息；此时 ring 为空，head 和 tail 都在 24。
3. 写入 40 字节 payload，占 48 字节，需要先跳过尾部的 40 字节。
4. 原实现要求同时容纳 `40 + 48 = 88` 字节，但 ring 总容量只有 64。reader 已经消费完所有已发布内容，双方都无法推进；指定 timeout 时写入超时，不指定时会持续等待。

B 保留原有“填充和消息可以一起放入”的路径。空间不足以同时容纳二者、但能容纳尾部填充时，先发布填充标记并推进 head，再继续等待消息所需空间。reader 按原有协议消费旧消息和填充后推进 tail。writer 始终不修改 tail，且发布前检查填充确实落在空闲区域。

这样既解决等待条件永远无法满足的问题，也保留未读消息和背压。若发布填充后写入超时，消息尚未发布；reader 可以消费填充，随后重试仍能正常写入。现有 reader 和共享内存布局不变。

## A/B/C 实验

三个独立 worktree 位于仓库旁的 `agxinfer-worktrees/`，分支共同前缀为 `gw/framework-base/`。

| 角色 | 分支后缀 | 初始提交 | 初始 2 例 | 加强测试提交 | 最终 10 例（实测） |
| --- | --- | --- | --- | --- | --- |
| A：原始实现 | `shm-ring-wrap-baseline` | `3f5243f` | 2 failed | `0a4ac71` | 6 failed / 4 passed |
| B：正确修复 | `shm-ring-wrap` | `87defed` | 2 passed | `a5ef0d8` | 10 passed |
| C：对抗实现 | `shm-ring-wrap-adversarial` | `703b6a9` | 2 passed | `05b47ff` | 6 failed / 4 passed |

初始用例覆盖 ring 已空时的大消息回绕，以及占满整个 ring 的最大合法消息。A 实际读不到后续消息；B 能正常继续。

C 在 B 的填充发布路径中只增加一行生产代码：

```python
r.set_u64(_O_TAIL, head + skip)
```

它假装 reader 已消费完所有旧消息和尾部填充，靠丢弃未读数据腾出空间。ring 为空时没有未读数据，因此 C 成功通过全部初始用例。C 没有修改测试或识别测试名称。

加强用例先保留一条未读消息，再让下一条消息跨尾部，停止消费并等待 writer 的有限写入超时。检查未读消息原样保留、超时 payload 没有发布，以及恢复消费后重试只交付一次。C 被直接读到的错误 payload 检出：应收到 `unreadunreadunread!!`，实际收到后写入的 `zzzz...`。

其余用例检查：

- 发布填充后超时，再关闭 writer，旧消息仍须先于 EOF 交付。
- 64、65、67、128 字节 ring 上连续变长消息的字节内容和顺序；包括空消息、最大合法消息，以及不足 4 字节、无法容纳显式 marker 的尾部。
- 超大消息被拒绝，且不影响已排队的消息。

两个“停止消费再等待写入结果”的用例为确定性的 C 检出点。连续生产/消费用例有实际进程调度，表中的完整失败数量记录本次实测，不承诺每种错误实现或每次调度都产生相同失败数量。

A/B/C 最终测试文件逐字相同，Git blob ID 为 `692ebffaa37de0121c39c5369ed652e5b3372eea`。A 未改生产代码；C 相对 B 的生产代码只有上述一行。所有失败都是正常 pytest 断言失败，没有 `xfail` 或 skip。

## 验证边界

测试使用真实 ShmReader，以及独立 Python 子进程中的真实 ShmWriter。payload 只经过真实共享内存 ring；额外控制管道仅发出写入命令、报告完成或超时，不代替被测数据通路。测试不直接改游标、不 mock ring，也不依赖内存中声称“写入成功”的状态替代读出的数据。

环境为 macOS arm64 / CPython 3.12.13，使用仓库锁定依赖。B 的 pyright 为 0 errors，`git diff --check` 通过。

完整 IPC 测试：B 为 27 passed / 1 failed；基线的既有 IPC 用例为 17 passed / 1 failed。两者唯一失败都是 `test_slow_consumer_blocks_the_producer[tcp]`（基线为背压上界断言，B 为对端断开）。本次没有修改 TCP 实现，没有扩展到与此次修改无关的全项目测试，也没有处理其他 review 问题。

## 复现

在各自 worktree 中运行同一条命令：

```sh
uv run --frozen --python 3.12 pytest tests/ipc/test_shm_ring_wrap.py -q
```

B 应通过，A/C 应失败。检查生产代码差异：

```sh
git diff fface7e 0a4ac71 -- engine
# 无输出：A 只添加测试。
git diff a5ef0d8 05b47ff -- engine
# C 只多一行 writer 推进 tail 的代码。
```

初始实验保留了独立提交，可以在临时 worktree 复现 C 逃过弱测试的过程：

```sh
git worktree add --detach /tmp/agxinfer-shm-C-initial 703b6a9
cd /tmp/agxinfer-shm-C-initial
uv run --frozen --python 3.12 pytest tests/ipc/test_shm_ring_wrap.py -q
# 2 passed
```
