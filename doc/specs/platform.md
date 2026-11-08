# 规格：平台 `platforms.*`

- 类型：框架本体。跑在每个 Python 进程内部。改了它要重启引擎。
- 公共约定见 [README.md](README.md)。

## 1. 职责

把硬件差异挡在这一层后面，给单元提供：
1. 设备：有几张卡、型号、能力。
2. 显存：普通分配；预留虚拟地址再按需映射物理显存（给 KV 池 grow 用）。
3. stream 和 event。
4. graph 录制与回放的原语。
5. 平台信息（进身份）。

## 2. 平台

### 2.1 `cuda_sm100`（阶段 2 才实现）

- 基于 torch.cuda 和 CUDA VMM（`cuMemAddressReserve` / `cuMemMap`）。
- graph 原语包装 `torch.cuda.CUDAGraph`。

### 2.2 `fake`（阶段 1 实现，CI 用）

- 设备：可配置几张假卡。
- 显存：CPU 上的 buffer，每个 buffer 有一个假“地址”；grow 时已有地址不变。
- **假 CUDA graph**（骨架验证“绑定方式”的关键）：
  - 录的时候记下：用到的每个 buffer 的地址、shape、dtype（按地址），以及按值写死的东西（选中的 kernel solution id、标量参数、运行时标志）。
  - replay 时逐项检查：按值的东西变了，或者按地址的 buffer 换了地址、改了 shape / dtype，就报错 `StaleGraph`；只是同地址的内容变了，照常执行。
  - 不含任何模型语义：执行时只按记录的步骤调用假 kernel（按配置 sleep、按配置注入故障）。
- 平台信息：可配置，用来测试“两台机器平台不一致时拒绝加入”。
- 没有张量：`tensor()` 报 `TypeError`。

### 2.3 `torch_eager`（2026-11-08，阶段 2 的 eager 路径）

- 真张量，eager 执行，在 CPU 上或者进程自己的 CUDA 设备上跑。M2a 在 CPU 上，SM100 上的 eager 也用它；graph 留给 `cuda_sm100`。
- 显存：buffer 背后就是一个 torch 张量，`tensor(buf)` 交出来；地址是张量的 `data_ptr()`。分配时清零，和 sglang 的显存池一样。
- `reserve` / `map`：保留 VMM 的约定（基地址不变、view 只能落在已映射的前缀里），但预留时就把整段分配好，所以不省显存；设备能力里没有 `vmm`。
- 设备：CPU 是一个设备；CUDA 上是进程看得到的卡（节点代理按 `ProcSpec.gpus` 设 `CUDA_VISIBLE_DEVICES`）。
- stream：设备当前的 stream。单元不自己建 stream（stage2-contract.md §10.4）。
- `graph()`：报 `NotImplementedError`，不录 graph。
- 依赖 torch 2.11：CPU 用 uv 的 `cpu` extra，SM100 用 `gpu` extra（`pyproject.toml`）。

### 2.4 进程用哪个平台（2026-11-08）

- 拓扑里每个角色（`RoleSpec`）写 `platform`（默认 `fake`）和 `platform_options`。合法的名字在 `platforms.PLATFORMS`，`Topology.check()` 拒绝别的名字。
- `torch_eager` 的选项：`{"cuda": 1}` 用 CUDA，否则用 CPU，即使角色被放在了 GPU 上。
- 平台经 `DesiredProc`、`ProcSpec` 传给进程，进程启动时由 `make_platform` 建好。所以角色换平台，`core.deps` 和进程管家都重启它的进程。

## 3. 对外接口（概要）

```python
class Platform(Protocol):
    def devices(self) -> list[DeviceInfo]: ...
    def alloc(self, nbytes: int, device: int, *, shape=None, dtype="u8") -> Buffer: ...
    def free(self, buf: Buffer) -> None: ...
    def tensor(self, buf: Buffer) -> Any: ...   # torch 平台交出张量；fake 报 TypeError
    def reserve(self, nbytes: int, device: int) -> VirtualRange: ...   # grow 用
    def map(self, vr: VirtualRange, offset: int, nbytes: int) -> None: ...
    def stream(self, device: int) -> Stream: ...
    def synchronize(self, device: int) -> None: ...   # 等设备上已提交的工作做完（worker 的 quiesce）
    def graph(self) -> GraphRecorder: ...       # 只有 plan 单元能调用（静态检查）
    def info(self) -> PlatformInfo: ...
```

所有平台一致的约定（2026-11-08，PR #24 评审）：
- `alloc` 至少给 `max(nbytes, shape 所需字节)`；不给 `shape` 时，取 `nbytes` 里放得下的整数个元素，多出来的字节也在。
- buffer 的地址是实际分配的那段内存的起始地址，没有元素的 buffer 也有非零地址（`torch_eager` 至少分配一个元素；PyTorch 对没有元素的张量报 `data_ptr() == 0`，所以不用它）。
- 大小、偏移、维度都不能是负数（不支持 PyTorch 那种 `-1` 推断维度）；view 的偏移要按元素对齐。否则报 `ValueError`（`platforms.base.check_extent`、`check_view`）。
- 每个带设备号的调用都检查设备号。

## 4. 验收测试（阶段 1，fake 平台）

1. grow：映射新的物理区段后，已有 buffer 的地址不变，假 graph 仍然能 replay。
2. 假 graph：按值依赖变了不重录就 replay，报 `StaleGraph`；同地址改内容，正常 replay。
3. 换地址：buffer 重新分配后不重录就 replay，报 `StaleGraph`。
4. 只有 plan 单元能拿到 `graph()`：别的单元调用时静态检查报错。

`torch_eager`（2026-11-08，`tests/framework/platforms/test_platforms.py`）：
- 平台契约（分配大小、重复释放、负数和未对齐的参数、设备号、grow 和 view 的范围、stream、`synchronize`、设备、平台信息）在 `fake` 和 `torch_eager` 上跑同一组测试。
- buffer 是清零的张量，地址就是 `data_ptr()`；释放以后拿不到张量。
- grow 以后旧 view 和新 view 共用同一段内存。
- 拓扑里的角色写 `torch_eager`，它的进程里的单元从 `ctx.platform` 拿到真张量（`tests/framework/cluster/test_cluster.py`）；换平台会重启进程（`tests/units/core/test_deps.py`）。
