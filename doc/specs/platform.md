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

## 2. 两个平台

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

## 3. 对外接口（概要）

```python
class Platform(Protocol):
    def devices(self) -> list[DeviceInfo]: ...
    def alloc(self, nbytes: int, device: int) -> Buffer: ...
    def reserve(self, nbytes: int, device: int) -> VirtualRange: ...   # grow 用
    def map(self, vr: VirtualRange, offset: int, nbytes: int) -> None: ...
    def stream(self, device: int) -> Stream: ...
    def graph(self) -> GraphRecorder: ...       # 只有 plan 单元能调用（静态检查）
    def info(self) -> PlatformInfo: ...
```

## 4. 验收测试（阶段 1，fake 平台）

1. grow：映射新的物理区段后，已有 buffer 的地址不变，假 graph 仍然能 replay。
2. 假 graph：按值依赖变了不重录就 replay，报 `StaleGraph`；同地址改内容，正常 replay。
3. 换地址：buffer 重新分配后不重录就 replay，报 `StaleGraph`。
4. 只有 plan 单元能拿到 `graph()`：别的单元调用时静态检查报错。
