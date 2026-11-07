# 进程宿主每一步的框架开销：同步 vs async

- 日期：2026-11-07
- 对应：欠账“单元运行时同步还是 async：一直没测过框架开销”（框架本体第一块的决定是同步运行时，见 [../specs/unit-runtime.md](../specs/unit-runtime.md)）。
- 结论：**保持同步**。同步宿主每一步的框架开销约 3.7 µs；一个最小的 asyncio 循环光是“一次事件循环往返 + 一次异步 zmq 轮询”就要 8.2 µs，还没算真正的异步请求分发。真实的 decode 一步是毫秒级（5–50 ms），3.7 µs 不到它的 0.1%，改成 async 只会更慢，没有收益。

## 测的是什么

`ProcessHost.run()` 每一步做的事：
1. `safe_point()`：`SyncRpcServer.poll(0)`（zmq 零超时轮询，有请求就处理），再跑安全点钩子（现在只有变更参与者的 `tick()`，没有变更时立刻返回）；
2. `runtime.lookup("proc.loop@1")`：拿到循环单元的服务代理；
3. 通过代理调用 `step()`。

用一个什么都不做的 `step()`（计数加一）把框架开销单独量出来。另外量一个最小的 asyncio 版本作对照：每步 `await sock.poll(0)` + `await asyncio.sleep(0)`（让 RPC 处理之类的其他任务有机会跑）+ 直接调用 `step()`。这是 async 宿主每一步最少要多付的代价。

## 结果

机器：AMD EPYC 7763（128 线程），Python 3.12.3，pyzmq 27.3.0 / libzmq 4.3.5。每项 200000 步，跑 5 次取中位数，单位是微秒每步。

| 每一步 | µs |
|---|---|
| 直接调用 `step()` | 0.11 |
| 经过服务代理（`lookup` + 调用） | 0.61 |
| 只有 `safe_point()`（zmq `poll(0)` + 钩子） | 2.91 |
| `ProcessHost.run` 的循环体（以上全部） | 3.65 |
| 最小的 asyncio 循环（`await poll(0)` + `sleep(0)` + `step()`） | 8.21 |

- 开销大头是 zmq 的零超时轮询（约 2.9 µs）。真要省，可以每 N 步才轮询一次，或者只在 `step()` 很短时才这么做；现在没有必要。
- 服务代理每次调用约 0.5 µs，可以忽略。
- 这台机器有一张 RTX 4090 D（驱动 590.44.01），但还没装 torch，所以这里只量了 CPU 这一侧。GPU 上一步的时长会在阶段 1′ 的 GPU 探针里量，到时候可以用同一个脚本再核对一次占比。

## 脚本

在仓库根目录运行：`uv run python <脚本> [步数]`。

```python
"""Per-step framework overhead of the process host (sync), and what an async host would add.

Run from the repo root:  uv run python <this file> [steps]
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path[:0] = ["engine", "."]

import zmq
import zmq.asyncio

from core.common.ids import HostId, Identity, ProcId
from core.supervisor.protocol import ProcSpec
from tests.framework.support import spec, write_tree
from unit.runtime import ProcessPlan
from unit.runtime.host import ProcessHost

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200_000
REPEATS = 5
UNIT = '''
class Loop:
    def __init__(self, ctx, config):
        self.n = 0
        self.limit = 0
        self.host = None
        ctx.provide("proc.loop@1", self)

    def step(self):
        self.n += 1
        if self.n >= self.limit:
            self.host.stopping = True
        return self.n
'''
PROC = ProcId(HostId("bench"), "worker", 0)


def make_host(root: Path) -> tuple[ProcessHost, object]:
    loop = spec("bench.loop", "bench.loop:Loop", provides=["proc.loop@1"])
    plan = ProcessPlan(PROC, Identity("v1"), [loop], code_root=str(root))
    host = ProcessHost(ProcSpec(PROC, Identity("v1"), plan, "tcp://127.0.0.1:1", "127.0.0.1"))
    host.runtime.load_process_plan(plan).unwrap()
    host.paused = False
    return host, host.runtime.instance("bench.loop")


def per_step_us(fn: object, n: int) -> float:
    times = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(n)  # type: ignore[operator]
        times.append((time.perf_counter() - t0) / n * 1e6)
    return statistics.median(times)


def main() -> None:
    root = write_tree(Path(tempfile.mkdtemp()) / "engine", {"bench/loop.py": UNIT})
    host, inst = make_host(root)

    def bare(n: int) -> None:
        step = inst.step  # type: ignore[attr-defined]
        inst.limit = 1 << 62  # type: ignore[attr-defined]
        for _ in range(n):
            step()

    def proxy(n: int) -> None:
        inst.limit = 1 << 62  # type: ignore[attr-defined]
        for _ in range(n):
            host.runtime.lookup("proc.loop@1").step()  # type: ignore[union-attr]

    def safe_point_only(n: int) -> None:
        for _ in range(n):
            host.safe_point()

    def host_run(n: int) -> None:
        inst.n, inst.limit, inst.host = 0, n, host  # type: ignore[attr-defined]
        host.stopping = False
        while not host.stopping:  # ProcessHost.run's loop body, without the shutdown at its end
            host.safe_point()
            loop = None if host.paused else host.runtime.lookup("proc.loop@1")
            if loop is None:
                continue
            epoch = loop.step()
            if epoch is not None:
                host.epoch = epoch

    zctx = zmq.asyncio.Context()
    sock = zctx.socket(zmq.ROUTER)
    sock.bind("tcp://127.0.0.1:*")

    async def async_loop(n: int) -> None:
        step = inst.step  # type: ignore[attr-defined]
        inst.limit = 1 << 62  # type: ignore[attr-defined]
        for _ in range(n):
            await sock.poll(0, zmq.POLLIN)  # what replaces SyncRpcServer.poll(0)
            await asyncio.sleep(0)  # one event-loop turn, so other tasks (RPC handlers) can run
            step()

    def async_host(n: int) -> None:
        asyncio.run(async_loop(n))

    rows = [
        ("bare step()", per_step_us(bare, N)),
        ("step() through the service proxy (lookup + call)", per_step_us(proxy, N)),
        ("safe_point() alone (zmq poll(0) + hooks)", per_step_us(safe_point_only, N)),
        ("ProcessHost.run loop (all of the above)", per_step_us(host_run, N)),
        ("minimal asyncio loop (await poll(0) + sleep(0) + step)", per_step_us(async_host, N)),
    ]
    print(f"python {sys.version.split()[0]}, {N} steps x {REPEATS} repeats (median), microseconds per step")
    for name, us in rows:
        print(f"  {us:7.2f}  {name}")
    print("hooks at the safe point:", [getattr(h, "__qualname__", h) for h in host.safe_point_hooks])
    sock.close(0)
    zctx.term()
    host.rpc.close()
    host.client.close()
    host.zctx.term()


if __name__ == "__main__":
    main()
```
