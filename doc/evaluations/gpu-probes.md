# GPU 探针（阶段 1′）：真 CUDA graph、单元重载、显存扩容与契约

- 日期：2026-11-08
- 对应：[roadmap](../roadmap.md) 阶段 1′ 的三个问题和“契约探针”；[platform.md](../specs/platform.md)（假 graph 的绑定规则、grow）、[unit-runtime.md](../specs/unit-runtime.md)（卸载顺序、代码重载）、[state-changes.md](../specs/state-changes.md)（导出导入、排空）。
- 脚本和原始输出在 [gpu-probes/](gpu-probes/)：`run.sh` 依次跑完全部探针（P8 要 4 张卡，其余各要 1 张），`logs/` 是本文引用的输出。探针用完就扔，不进骨架。
- 本文只记录结论和建议。规格里要改的地方列在最后一节，规格文件本身这次没有改。

## 结论

1. **问题 1（卸掉、重新 import、装上以后，decode graph 能不能正确重录和 replay）：能。** 玩具单元连续 30 轮、fla 的 GatedDeltaNet decode 连续 10 轮、flashinfer 的 paged decode 连续 4 轮，每轮都正确。
2. **问题 2（按规定顺序卸载，会不会读到已释放的显存）：按规定顺序不会；顺序一错，真 graph 不报错，悄悄读到垃圾数据。** 所以“先卸使用方、再卸持有方”必须由运行时保证，`cuda_sm100` 平台也要像假 graph 一样在 replay 前自己检查依赖。
3. **问题 3（同一进程反复重载会不会留下撤不掉的东西）：有三处，都有明确的办法。**
   - `torch.library` 的注册不随模块消失，不撤销就重新 import 会报“重复注册”。
   - 每个用过 cuBLAS 的 stream 各缓存一份 workspace（约 8 MB），PyTorch 的 stream 池有 32 个，所以最多涨约 270 MB 就停。全程复用一个 stream 就完全不涨；`torch._C._cuda_clearCublasWorkspaces()` 能全部还回去。
   - graph 删了但输出 tensor 还被引用着，它的私有显存池就不会释放。

   另外，fla 和 flashinfer 反复卸载、重新 import，显存和注册的算子数都不再增长。
4. **假 graph 的绑定模型和真 graph 一致，但真 graph 还依赖三样假 graph 没有表达的东西：**
   - kernel 代码：模块卸掉、引用全丢以后，旧 graph 照样跑旧 kernel，不报错；
   - view 的 shape：graph 里写死，后来扩出来的显存它看不见；
   - 共享显存池：旧 graph 不按录制顺序回放，会覆盖新 graph 的输出。
5. **VMM 扩容（grow）可行：** 地址不变，扩容前录的 graph 能读到新映射的页，不用重录、不用排空，也不等其他 stream 上正在跑的 kernel。每 1 GiB 约 0.4–5 ms。
6. **契约探针通过：** sglang 的 attention backend 接口和 Qwen3.8 的混合状态（paged KV + 每请求一个 GDN 槽位）能装进单元接口。在真 kernel 上，同一份调度分别用 eager 和 graph replay 跑 160 步，token 和 logits 逐位一致。graph 那一遍中途还做了三件事：
   - KV 池扩容两次，没有重录；
   - 原地 refit 一次权重；
   - 16 个请求都在途时换了一次 backend 代码，没有排空，重录耗时 23–104 ms。
7. **契约要改两处：**
   - replay 前的检查不能逐个 tensor 比对：55 项依赖每步约 38 µs，真模型会有几百项。
   - capture 前的 warm-up 会真的改状态，状态单元要给 plan 单元提供 scratch 槽位。
8. **代码重载不能原地改写文件。** 同一秒内重写、长度不变的 `.py` 会 import 到旧字节码。引擎现在的做法刚好避开了：每个身份一个快照目录、`PYTHONDONTWRITEBYTECODE=1`、change-set 排除 `__pycache__`。这一条应当写进契约。
9. **工具链：** fla 0.6.0 的 chunk prefill 在 torch 2.8 / Triton 3.3.1 上 autotune 挂住，换到 torch 2.11 / Triton 3.6 就正常。v0 的 Qwen3.8 工具链应当按后者这一档选。
10. **TP 和 NCCL：**
    - 含 NCCL all_reduce 的 decode 步可以录进 graph 并正确 replay，原地 refit 以后也正确。
    - 同一个进程里反复新建、销毁通信器 8 次，显存和线程数都不增长。
    - 在同一组进程里从 TP4 换到 TP2（新建通信器、重新切权重、重录）只要约 1 s。停掉旧进程、起新进程要 8–16 s，主要花在 CUDA context（4–10 s）和 NCCL 初始化（2–4 s）上，这还没有算真实权重的加载时间。
    - 所以 TP 变更不一定非要重启 rank 进程，见下面“对契约的影响”第 11 条。

## 环境

- 一台 8 卡 SM100 级节点，卡间全互联 NVLink。P8 用 4 张卡，其余探针各用一张，用的都是空闲卡（`nvidia-smi` 上没有进程、显存为 0）。PyTorch 报告 capability 10.0、148 SM、约 178 GiB 可用。驱动 580.105.08，CUDA 13.0，Python 3.12.3。
- 第一轮的 P1、P4 输出里带有设备名，提交进 `logs/` 时删掉了这个字段；脚本现在改为输出 capability 和 SM 数。
- 两套软件栈。P1、P2、P7 只在 A 上跑过，其余在 A、B 上各跑一遍：

| | A：容器自带 | B：节点上已有、跑过 Qwen3.8 vLLM 服务的 venv |
|---|---|---|
| torch | 2.8.0a0+34c6371d24.nv25.08 | 2.11.0+cu130 |
| Triton | 3.3.1 | 3.6.0 |
| cuda-python | 13.0.0 | 13.3.1 |
| fla | 0.6.0（`pip install --no-deps --target` 到临时目录） | 同左 |
| flashinfer | 0.7.3 + apache-tvm-ffi 0.1.15（同上） | 0.6.14（venv 自带） |
| NCCL（torch 自带） | 2.27.7 | 2.28.9 |

- B 只用它的 Python 解释器：设置了 `PYTHONDONTWRITEBYTECODE=1`，Triton 和 flashinfer 的缓存目录都指向自己的临时目录，没有改动这个 venv 和它的缓存。
- 对照用的 sglang 源码是 0.5.23 的 wheel，只读。

## 各探针的结果

### P1 真 CUDA graph 依赖什么（`p1_graph_binding.py`，A）

| 录制后做了什么 | replay 的结果 |
|---|---|
| 权重原地 `copy_` 新值 | 读到新值 |
| 输入 buffer 改内容 | 读到新值 |
| 改录制时用到的 Python 标量 | 仍然用旧值（录制时写死了） |
| 把引用换成另一个地址的 tensor | 仍然读旧地址，不报错 |
| 释放 graph 依赖的 buffer，再让别的数据占用这块显存 | 读到垃圾，不报错 |
| graph 里的 Triton kernel，原地改它的输入 | 正确 |

这次输出里的 `pool_mb` 不能直接读成“删了 graph 还剩约 100 MB”。原因有两个：
- 脚本里的变量 `_` 还引用着第二个 graph 的输出（64 MiB），所以它的私有池没有释放；
- 两个 capture stream 各有一份 cuBLAS workspace。

P3 把这两件事分开重测了。

### P2 同一进程反复重载一个单元（`p2_unit_reload.py`，A）

测试单元在 import 时注册一个 `torch.library` 算子、建一个模块级 stream、定义一个 Triton kernel。

| 模式 | 结果 |
|---|---|
| `destroy`：卸载时调用 `LIB._destroy()` | 30 轮全对。第一轮重录 728 ms，之后每轮 29 ms，卸载 26–100 ms。allocated 从 48.9 MB 涨到 170.8 MB，reserved 从 78 MB 涨到 378 MB |
| `keep`：不撤销注册 | 第 2 轮重新 import 时报 `RuntimeError: Tried to register an operator ... multiple times` |

显存为什么涨，在 P3 查清了：这个单元自己也建了一个 stream，所以每轮从 stream 池里取两个，用来 warm-up 的只有一半，30 轮里实际碰到 16 个不同的 warm-up stream。

### P3 显存残留的来源（`p3_residue.py`，A、B 结果完全相同）

每种模式 80 轮，每轮做一遍“warm-up、capture、replay、核对、全部删掉”，然后读 caching allocator 的统计。

| 模式 | 用到的 stream 数 | allocated（MB），第 0 轮 → 第 79 轮 | reserved（MB） | 第 32 轮以后的增长 |
|---|---|---|---|---|
| `p2_like`：每轮新建一个 stream 做 warm-up（和 P2 一样） | 32 | 48.9 → 292.6 | 78 → 678 | 0 |
| `p2_like_clear_ws`：同上，卸载后清掉 cuBLAS workspace | 32 | 24.5 → 24.5 | 38 → 38 | 0 |
| `p2_like_no_cublas`：同上，但 graph 里没有 matmul | 32 | 24.5 → 24.5 | 38 → 38 | 0 |
| `shared`：warm-up 和 capture 全程用同一个 stream | 1 | 40.8 → 40.8 | 58 → 58 | 0 |

- **增长全部来自每个 stream 一份的 cuBLAS workspace**，前 32 轮里每多一个 stream 约多 8 MB。PyTorch 的 stream 池每种优先级 32 个、循环复用，所以碰满 32 个以后就不再涨。
- 碎片（inactive split）从 29 MB 涨到 385 MB，也是这些 workspace 散落在各处造成的。清掉 workspace 或者全程用一个 stream，碎片都不涨。

graph 私有池（`pools` 模式，数值是相对基线的增量，单位 MB；两个 4096×4096 fp32 输入，各 64 MiB）：

| 状态 | allocated | reserved |
|---|---|---|
| 录 g1：`(a@b).relu()` | 64 | 130 |
| 再录 g2，各用各的池 | 128 | 258 |
| 删掉 g1，但还留着它的输出 o1 | 128 | 194 |
| 再删 o1 | 64 | 130 |
| 删掉 g2 和 o2 | 0 | 0 |
| g3、g4 共享一个池 | 128 | 194 |
| 全部删掉 | 0 | 0 |
| 清掉 cuBLAS workspace | −8.1 | −20 |

**共享池有正确性风险。** g5、g6 共享一个池，按 g5、g6 的顺序录制，录 g6 之前把 g5 的输出删掉了，于是 g6 的输出落在了 g5 原来的输出上（`overlaps = true`）。之后先回放 g6、再回放 g5，**g6 的输出被 g5 覆盖**（`clobbered = true`）。在后台准备新版本时，旧 graph 还在回放，新 graph 正在录制，正好就是这个顺序。

### P4 用 VMM 给 KV 池扩容（`p4_vmm_grow.py`）

先预留 64 GiB 虚拟地址，按 1 GiB 一块映射物理显存（`cuMemCreate`、`cuMemMap`、`cuMemSetAccess`），再通过 `__cuda_array_interface__` 包成 torch tensor。

| 检查项 | A | B |
|---|---|---|
| 映射粒度 | 2 MiB | 2 MiB |
| 映射 1 GiB 耗时（第一次 / 第二次 / 另一个 stream 正忙时） | 0.99 / 5.21 / 0.33 ms | 0.38 / 0.36 / 0.42 ms |
| 映射 8 GiB；解除映射 8 GiB | 0.94 ms；3.6 ms | 6.08 ms；10.21 ms |
| 映射后设备空闲显存减少 / torch 的 reserved 是否变化 | 1024 MiB / 不变 | 同左 |
| 扩容前后基址不变 | 是 | 是 |
| 扩容前录的 graph（Triton kernel 按“基址 + 页号”访问）能读到新映射的页 | 是 | 是 |
| 扩容前录的 graph（对整个 view 做 `sum`）只算原来那一段 | 是 | 是 |
| 映射时另一个 stream 上的 kernel 还没跑完（映射没有等它） | 是，还剩 158 ms | 是，还剩 147 ms |
| 解除末尾没用到的映射后，graph 仍然正确，显存全部还回 | 是 | 是 |
| 只剩 view 引用时，持有方对象仍然活着；最后一个 view 释放后才解除映射 | 是 | 是 |

没有故意去测“解除映射以后再回放 graph”。那样会触发 GPU 页错误，在共享节点上会留下 Xid 记录。

### P5 Qwen3.8 用到的 kernel 库反复重载（`p5_kernel_reload.py`）

形状按 Qwen3.8-27B 的 TP1：
- GatedDeltaNet：16 个 key 头、48 个 value 头，head dim 128，batch 32，递归状态放在一个 64 槽的池里。
- full attention：24 个 q 头、4 个 kv 头，head dim 256，page 16，batch 16。

每轮的做法：把 `fla` 或 `flashinfer` 从 `sys.modules` 里全部移走，再 import，然后 capture 并 replay，结果和 fp32 参照实现比对。

| 模式 | A（torch 2.8 / Triton 3.3.1） | B（torch 2.11 / Triton 3.6） |
|---|---|---|
| `fla`：decode 重载 10 轮 | 全对。import 第一轮 2.9 s、之后 0.77 s；第一次调用第一轮 991 ms（编译）、之后 6–7 ms；capture 87–186 ms。注册的算子数固定为 +52，allocated 固定为 205.8 MB | 全对。import 3.6 s / 1.1 s；第一次调用 638 ms / 7–118 ms；capture 0.2–5 s，波动很大，原因没查。算子数固定为 +62，allocated 固定为 205.8 MB |
| `fla_prefill`：第一次 chunk 调用 | **挂住**。autotune 在 `chunk_fwd_o` 两个候选 config 里的第一个上，跑了 15 分钟以上没有返回，GPU 一直 100%，用 py-spy 看到的栈见 `logs/p5_kernel_reload_run1_hang.log`。第二次跑到 600 s 超时（见 `logs/p5_kernel_reload.log`） | 正确。第一次调用 28.9 s（编译加 autotune），第二次 1.4 ms |
| `fla_stale_graph`：capture 后把 fla 全部移出 `sys.modules`、丢掉所有引用、gc，再回放旧 graph | 旧 graph 仍然正确执行，fla 模块剩 0 个 | 同左 |
| `flashinfer`：重载 4 轮，JIT 缓存目录一开始是空的 | 全对。第一轮 plan 加第一次运行 19.6 s（JIT 编译），之后 25–27 ms。import 0.5 s / 0.3 s，capture 55–90 ms，重新 plan 0.2–0.4 ms。算子数固定为 +5，allocated 固定为 648.7 MB | 全对。缓存目录已被 P6 预热，第一次 plan 加运行 72 ms，之后 18–24 ms。import 0.7 s / 0.11 s，重新 plan 0.3 ms。算子数固定为 +2，allocated 固定为 648.7 MB |

flashinfer 那一行两边的结论相同：
- 用短序列（1–64）的 plan 录 graph，换成长序列（1000–4000）重新 `plan` 后直接回放，结果正确，并且和同一个 wrapper 的 eager 输出逐位相同。
- 每轮重新 import 以后，显存和注册的算子数都不增长，也不再重新编译（第一次运行 18–27 ms）。

B 的 `fla_stale_graph` 第一遍统计 fla 模块数的条件写错了（把 `flash_attn` 等也算了进去），修正后单独重跑，见 `logs/p5_kernel_reload_B_stale_graph_rerun.log`。

### P6 契约探针：截断的 Qwen3.8 decoder（`p6_contract.py`）

8 层，排列和 Qwen3.8 一样：3 层 GatedDeltaNet 加 1 层 full attention，重复两次。头数和 head dim 用 Qwen3.8 的真实值，hidden 1024，词表 4096，batch 16。按单元来拆：

| 单元 | 种类 | 内容 | 来源 |
|---|---|---|---|
| `kv.paged` | 状态 | 每个 attention 层的 K、V 各占一段 VMM 区间。view 覆盖整个预留区间，物理显存按 2 MiB 的整数倍映射，所以第 0 页可以留作 scratch 页 | sglang `KvVmmArena` / `KvVmmBufferOwner.ensure_prefix`（0.5.23 已经用 VMM 给 KV 扩容） |
| `gdn.slots` | 状态 | 每个 GDN 层一份 conv state `[槽, 10240, 3]` bf16 和递归状态 `[槽, 48, 128, 128]` fp32，每个请求固定占一个槽，最后一个槽留作 scratch 槽 | sglang `MambaPool` / `HybridReqToTokenPool` |
| `weights` | 状态 | 全部权重 | |
| `backend` | 逻辑，中途换代码 | attention：flashinfer decode。每步在 graph 外算好 page 表并调用 `plan`，结果写进固定地址的 buffer；graph 内只做 KV 写入和 `run`。GDN：conv 更新加 fla 的 `fused_recurrent`，按槽号 gather/scatter | sglang `AttentionBackend` 的 `init_forward_metadata_out_graph` / `_in_graph` 分工 |
| `plan` | 逻辑，录 graph | capture 时记下 52 个按地址的依赖和 3 个按值的依赖（backend 版本、backend 实例、batch 大小），每次 replay 前逐项检查 | |

同一份请求调度（每个请求长 8–200 个 token，跑完就换新请求）先 eager 跑一遍，再用 graph replay 跑一遍。graph 那一遍：
- 页不够时给 KV 池扩容，不重录；
- 第 50 步原地 refit `lm` 权重，不重录；
- 第 90 步 16 个请求都在途时换 backend 代码，新代码里 KV 写入改用 `index_put_`，换完重录。

| 结果 | A | B |
|---|---|---|
| 160 步的 token 一致 / logits 逐位一致 | 是 / 是 | 是 / 是 |
| KV 池扩容次数和耗时（每次 4 段 × 4 MiB） | 2 次：49.8、4.7 ms | 2 次：0.5、16.4 ms |
| 换代码时在途的请求 / 换代码加重录的耗时 / 换完后的版本号 | 16 / 104 ms / 2 | 16 / 23 ms / 2 |
| 换代码后旧 graph 被检查出过期的依赖 | backend 的 5 个元数据 buffer、GDN 的槽号 buffer、`backend.version`、`backend.instance` | 同左 |
| 把 `kv.k0` 换成新分配的 tensor（模拟一次不用 VMM 的扩容）后被检查出的依赖 | `kv.k0` | 同左 |
| replay 前检查 55 项依赖的耗时（中位数） | 39 µs | 38 µs |
| 每步耗时中位数：graph / eager | 1.61 / 4.65 ms | 1.53 / 4.41 ms |

**第一遍 A 上出现过一个问题。** 当时换代码的做法是原地改写同一个文件，结果过期依赖里没有 `backend.version`：import 到的其实还是旧代码，见 `logs/p6_contract_run1_inplace_rewrite.log`。改成每个版本一个目录以后就正常了。原因由 P7 确认。

### P7 原地改写文件后重新 import（`p7_stale_bytecode.py`，只用 CPU）

先后写入 `VERSION = 1`、`2`、`3`，前两次在同一秒内，第三次隔 1.1 秒，每次写完都移出 `sys.modules` 并重新 import。三次实际 import 到的是 `1, 1, 3`。原因是 CPython 判断 `.pyc` 是否过期只看源文件的 mtime（整秒）和大小。关掉字节码缓存以后，import 到的是 `1, 2`。

### P8 TP 与 NCCL（`p8_tp_nccl.py`，4 张卡）

模型是一层 TP 的 MLP：hidden 4096，ffn 16384，batch 32，bf16。W1 按列切，W2 按行切，最后做一次 all_reduce。参照结果用完整权重在单卡上算。

A、B 的两次完整运行是在同一台机器上同时跑的，各用 4 张卡，CUDA context 和 NCCL 的初始化会互相拖慢。所以又让 B 在机器上只有它一个任务时单独跑了一遍 restart。A 的 c10d 会打印 `localhost` 的 IPv6 连接警告（errno 97），不影响结果。

**inproc：4 个 rank 进程从头到尾不退出**

| 检查项 | A（NCCL 2.27.7） | B（NCCL 2.28.9） |
|---|---|---|
| CUDA context 创建（4 个进程同时） | 12.4–12.9 s | 12.6–13.0 s |
| `init_process_group` 加第一次 all_reduce | 6.0–6.4 s | 5.2–5.5 s |
| 每个 rank 上通信器占的显存（按设备空闲显存的变化算） | 约 1060 MiB | 0.4–1.1 GiB |
| 含 all_reduce 的 graph：capture 后 replay | 正确，相对误差 5.2e-3（bf16 的量级） | 同左 |
| 原地 refit W2 后直接 replay | 正确 | 同左 |
| “新建通信器、capture、replay、删 graph、销毁通信器”循环 8 次 | 每次都正确。新建加第一次 all_reduce 1.7–2.7 s，capture 34–59 ms。空闲显存不减少，线程数固定在 262–263 | 每次都正确。2.3–3.6 s，capture 4–11 ms。同左 |
| 同一组进程里从 TP4 换到 TP2：给 rank 0、1 新建通信器，按 TP2 重新切权重，重录 | 新建加第一次 all_reduce 0.73–0.75 s，重新切分并拷到卡上 89–92 ms，capture 42–46 ms。结果正确，和 TP4 的输出不是逐位相同 | 0.78–0.81 s，65–105 ms，4.5–45 ms。同左 |
| 按“先使用方、后通信器”的顺序拆掉 TP4：先删 graph、输出和分片，再销毁通信器 | rank 2、3 的空闲显存回来 458–714 MiB，TP2 的 graph 仍然正确 | 442–698 MiB，同左 |

**restart：停掉 4 个 rank 进程，再起 2 个新进程跑 TP2**

这就是现在的动作推导对 TP 变更给出的做法。权重从本地盘 mmap 读，只有 2 × 128 MiB。下表是每个 rank 的耗时，格式为“TP4 / TP2”：

| 阶段 | A | B | B 单独跑 |
|---|---|---|---|
| spawn 到进入 main（含 import torch） | 1.4 / 1.3 s | 1.4 / 1.3 s | 1.3 / 1.3 s |
| CUDA context | 9.2 / 6.6 s | 10.0 / 8.2 s | 5.3 / 4.1 s |
| NCCL init 加第一次 all_reduce | 4.1 / 2.2 s | 4.1 / 2.2 s | 2.5 / 1.8 s |
| 读权重、切分、拷到卡上 | 0.06–0.17 / 0.07–0.09 s | 0.06–0.12 / 0.08–0.09 s | 0.07–0.08 / 0.16–0.17 s |
| capture | 0.19–0.32 / 0.21–0.23 s | 0.19–0.23 / 0.18–0.21 s | 0.17–0.19 / 0.72–0.75 s |
| **spawn 到第一步算对** | **15.2 / 10.5 s** | **16.0 / 12.1 s** | **9.4 / 8.2 s** |

作为对照，机器空闲时单个进程在单卡上创建 CUDA context：A、B 各连续测 3 次，在 0.85–3.6 s 之间；另外 A 有一次单独测到 9.7 s。

## 对契约的影响（建议，规格还没改）

1. **`cuda_sm100` 平台要做 replay 前检查，但不能逐个 tensor 比对。**
   - 硬件不提供任何保护（P1）。旧 graph 在模块卸掉以后照样能回放，跑的是旧 kernel（P5 `fla_stale_graph`）。
   - 逐个比对 55 项每步约 38 µs，占一个 1.5 ms 小模型 decode 步的 2.5%。64 层的真模型依赖会上百项。
   - 建议改成每个状态单元、每个提供缓冲区的逻辑单元各维护一个**绑定代数**：只要换了地址、shape 或 dtype，代数就加一。replay 前只比对这些代数和少量按值依赖。逐个 tensor 比对只在测试和调试模式下打开。
   - 代码版本（单元实例 id 或身份）要作为按值依赖记进 graph。
2. **capture 用的 stream 由平台提供，单元不能自己建 stream。** warm-up 和 capture 用同一个 stream（P3 `shared`）。单元运行时在卸载时调用 `_cuda_clearCublasWorkspaces()` 兜底。静态检查可以禁止单元调用 `torch.cuda.Stream()`。
3. **`torch.library` 注册必须通过 `ctx.effect` 登记撤销函数**（P2）。这已经符合 unit-runtime.md §5，可以加一条植入缺陷测试。
4. **graph 的输出也是 graph 的一部分。** plan 单元释放 graph 时要连输出一起放掉，否则私有池不会释放（P3）。新版本的 graph 不能和仍可能回放的旧 graph 共享池（P3 的覆盖问题）。所以后台准备期间的显存峰值等于新旧两套池之和，这要算进 4.7 “后台准备不了就推迟到排空之后”的判断里。
5. **`grow` 的契约：**
   - 状态单元的 view 覆盖整个预留区间，物理映射按 2 MiB 粒度增长，地址不变（P4、P6）。
   - 只有“按基址加页号访问”的 kernel 才能看到新映射的页；graph 里写死的 view shape 是按值依赖，所以 view 一开始就要按预留区间的大小建（P4）。
   - grow 不会影响在途请求，也不等其他 stream 上的 kernel，不需要排空，和 v0-plan 4.5 的推导一致。
   - state-changes.md §6.1 的“teardown 只放引用”在 VMM 上同样成立：最后一个 view 释放后，持有方才解除映射（P4）。
6. **状态单元要提供 scratch 槽位或 scratch 页给 plan 单元 capture 用。** capture 前的 warm-up 会真的执行：GDN 的递归状态会被推进，KV 会被写入。运行中途重录（换代码以后）如果用在途请求的槽位来 warm-up，就会把它们的状态弄坏。sglang 用假 batch 来 capture，原因也在这里。这是 P6 里唯一需要增加的状态接口。
7. **attention backend 的单元接口可以直接用 sglang 的分工：**
   - graph 外：每步的 host 端准备，在原地写进本单元的固定 buffer。这些 buffer 是 graph 的按地址依赖。
   - graph 内：形状固定的 GPU 操作。
   - flashinfer 在 cuda-graph 模式下重新 `plan` 以后可以直接回放，同一 batch 大小内不同序列长度都正确（P5、P6）。所以 plan 的结果不是按值依赖，batch 大小才是。
8. **代码重载的前提：新版本的代码在新路径下，绝不原地改写。** 快照目录里不能有 `.pyc`（P7）。`core.nodeagent` 现在设了 `PYTHONDONTWRITEBYTECODE=1`，`core.control.client` 和 `core.change.blobs` 也排除了 `__pycache__`，这几处要作为契约固定下来，并补一条测试。
9. **Qwen3.8 的状态规模：**
   - 每个请求在每个 GDN 层要 3 MiB 递归状态（48×128×128×fp32）加 60 KiB conv state。TP1 下 48 层合计约 147 MiB，256 个并发请求就是约 37 GiB。所以 `gdn.slots` 是一个固定槽位的池，大小要在启动时根据显存来定，也可以用同样的 VMM 办法扩槽位。
   - config 里有 1 层 MTP（`mtp_num_hidden_layers = 1`），v0 的投机解码草稿可以用 MTP。
10. **工具链：** v0 的 Qwen3.8 后端按 torch 2.11 / Triton 3.6 这一档来选。fla 0.6.0 在 Triton 3.3.1 上的 prefill 会挂住（P5）。
11. **TP 变更可以不重启进程。** v0-plan 4.5 现在对 TP4 → TP2 推导出的是“重建所有 rank 进程”。P8 说明还有一条便宜得多的路：
    - rank 进程和 CUDA context 都留着；
    - 通信器作为派生产物重建：新建、第一次 all_reduce 约 0.8 s；
    - 权重状态单元按新的切法重新加载；
    - 依赖旧通信器的 graph 重录。

    启动开销大约是重启进程的十分之一。建议这样落到规格里：
    - 资源依赖图里加入“通信器”这类派生产物。graph 对它是按值依赖，拆除顺序是先 graph、后通信器。
    - 重启进程只作为兜底。
    - 新旧两个通信器会同时存在，每个 rank 每个通信器约 0.4–1 GiB，要算进准备阶段的显存峰值。
    - 退出 TP 的 rank 拆掉 graph、分片和通信器以后，显存就还回来了。
    - 有一点限制要注意：c10d 的 `new_group` 只能在最初的进程集合里建组，所以扩大 world（例如 TP2 → TP4 要加入新进程）不能这样做，需要新的 rendezvous 或者直接用 NCCL 的 unique id 建通信器。这种情况没有测。

## 没有覆盖的

- prefill 没有用 graph，P6 的新请求都从 1 个 token 开始，不经过 prefill。
- 没有用真实权重，也没有 RoPE。
- TP 只测了单机 NVLink 上的一层 MLP。没有测：
  - 扩大 world；
  - 跨机；
  - custom all-reduce；
  - 销毁通信器以后再回放依赖它的 graph（这和 P1 的“先释放持有方”是同一类错误，预计会挂住或出错，没有故意去触发）。
- 没有测 GDN 状态在块边界打快照（前缀缓存要用），也没有测 MTP。
- 只用了一台机器，每项只跑了一两次，耗时数字只能看量级。
- 没有测“VMM 解除映射后再回放 graph”的故障形态（原因见 P4）。
- B 环境下 fla decode 的 capture 耗时在 0.2–5 s 之间波动，原因没有查。

## 复现

```sh
pip install --no-deps --target /tmp/probe_deps fla-core flashinfer-python apache-tvm-ffi   # B 环境只需要 fla-core
GPU=0 GPUS=0,1,2,3 DEPS=/tmp/probe_deps bash doc/evaluations/gpu-probes/run.sh            # 输出写到 gpu-probes/logs/
```
