# DS 1017 上多卡：DP 加 EP、EPLB、融合 kernel，整个模型出文本（2026-11-15）

- 记录人：Claude Fable 5.2（Claude Code）。
- 对应：[../specs/v1-ds1017.md](../specs/v1-ds1017.md) 第 10 节；PR #66（草稿，没合）。前一份记录：[2026-11-14-ds1017-production-kernels.md](2026-11-14-ds1017-production-kernels.md)。
- 机器：本地 4090 D（CPU 测试、两个 worker 共用一张卡）；SM100 节点 n1 的 4 张卡（任务服务，DeepInfer 的环境）。
- 2026-11-15 下午用户接手继续开发，这份记录写到交接时为止。没做完的见最后一节。

## 结论

1. **整个 DeepSeek-V4.1-Flash-1017 在 4 张 SM100 上经引擎的 HTTP 接口生成了有意义的文本。** 官方算术（后端 `tilelang`）和 DeepInfer 的融合 kernel（后端 `deepinfer`）两条路都出了；英文、中文问答的续写和 DeepInfer 在同一台节点上给出的前三四问逐字相同。
2. **多 rank 和单 rank 算的是同一个函数**（参照和 `tilelang` 后端）：本地 CPU、4090 D、SM100 的真实形状和真权重前 8 层上，几个 rank 的文本和一个 rank 逐字相同。
3. **EPLB 做了，不改变结果**：整个模型上一轮里重排 15 次，最忙的 rank 从平均值的 1.34 倍降到 1.05 倍，文本和不开时逐字相同。
4. **融合 kernel 接了 MoE 和 Hyper-Connections 这半边**（`mega_mhc`、`bf16_mega_gate`、带共享专家的 `fp8_fp4_mega_moe`），带 graph：
   - 单卡真权重 8 层一步从 4.5 ms 到 3.0 ms；
   - 4 张卡整个模型，每卡 64 行时一步约 32.8 ms，DeepInfer 是 18.8 ms（约 57%）。
   - attention 那半边和 Engram 还没融合，是剩下的差距。
5. **改了一处框架本体**（`core.change`：同一个角色的进程一起启动），原因和验证见下。没有写 framework-bugs 报告：没有发现更新机制的问题，但这次也几乎没有在跑着的引擎上热改（见「怎么做的」）。

## 做了什么

| 部分 | 内容 |
|---|---|
| 调度域（`core.sched`） | `sched.domains`：各 rank 跑各自的请求，每个域自己的行、账本（`ledger.domains@1`）、compose 视图；所有域一起走，空闲的域收空计划；一个域拒一步不影响别的域 |
| 通信组（`worker.comm`） | `torch.distributed`，照 DeepInfer：gloo 的默认组，卡上另建 NCCL 组；`gather` 和 `reduce` 两个 collective 给参照 kernel 用 |
| 专家和 Engram 表按 rank 分 | 模型按 rank 声明和装载自己的专家、表的行；`ds.ep_experts@1`、`ds.engram_fetch@1`（`kernels/ds1017_ep.py`）；空闲的 rank 每层都参与（`model.idle`） |
| EPLB | `policies.eplb`（决定，照 DeepInfer 的贪心）加 `state.expert_placement`（映射、计数、搬权重） |
| 融合 kernel | 组合 definition `ds.hc_step@1`、`ds.moe@1`；模型循环按 DeepInfer 的顺序重排；`kernels/deepinfer/mega.py` |
| 前后的钩子 | `Backend.fetch`（forward 之前、graph 之外：多 rank 时取 Engram 的行）、`model.stepped`（forward 之后：EPLB 计数） |
| 部署、前端 | `rsinfer serve --gpus 0,1,2,3` / `--ranks N`；`chat_template.jinja`、模板参数、BOS |

## 怎么做的，和 AGENTS.md 的出入

AGENTS.md 要求默认在跑着的引擎上热改。这次**大部分不是这样做的**，写明原因：

- **调度域**是 `core.sched` 的逻辑，用调度测试的夹具（真调度器加夹具 worker）最快，改一次跑 `tests/units/scheduler` 十几秒。原有的 79 个测试一个没改地通过，另加 14 个。
- **多 rank 的整条路**每次都要起一个新的多进程引擎（通信组的大小是进程级的，换了要换进程），所以是冷启动：本地用 `tests/scenarios/test_ds1017.py` 里两个 rank 的那个测试（20 秒一次）和 `tests/gpu/check_ds1017_ranks.py`。
- **SM100 上**没有用跟分支的任务，都是单个任务：整个模型冷启动一次 2–5 分钟（页缓存热的时候），4 张卡的任务前后排队，一共提交了十几个任务。
- 所以这次**没有检验热更新机制本身**。按设计应该成立、但没有试过的：
  - 多 rank 引擎上 `apply --config '{"dsv4.backend": ...}'`（专家权重要就地换行顺序，`experts.layout` 标记）；
  - 重载 `worker.comm`（卡上的组销毁重建，依赖它的单元跟着重载）；
  - 重载 `kernels.deepinfer`（共享缓冲区销毁重建，要所有 rank 同时）。

## 框架改动：同一个角色的进程一起启动

- **故障**：变更的协调者一个一个地启动进程，等前一个 check in 再起下一个。通信组的 rank 在加载单元时互相等，第一个 worker 永远等不到第二个，引擎起不来（`supervisor.no_check_in`）。
- **改动**（`engine/core/change/coordinator.py`）：同一个角色的进程用 `asyncio.gather` 一起启动；角色之间仍然一个一个来，重启仍在新建之前。每个角色一个进程时行为不变。
- **单元这一侧为什么绕不过去**：把碰头推迟到第一步可以让进程先 check in，但 graph 是单元加载时录的，带 collective 的 graph（`mega_moe`）必须所有 rank 同时录。
- **验证**：框架测试和全量测试通过；formal 全流程（TLC 加轨迹回放）59 个通过，用的是临时下载的 Temurin JRE 17。

## 查到的几件事

- **官方的算术不是 batch 无关的**：`tilelang` 后端下，几个请求一起问和单独问文本不同，单 rank 上就是这样。所以"多 rank 对单 rank"只比单独问；参照 kernel 是 batch 无关的，一起问也相同。整个模型（训练好的）不混沌，单独问和一起问相同。
- **逐位相同是怎么保住的**：每个 rank 把每个 token 的每一项单独送回，不先求和；token 所在的 rank 按专家编号的顺序加。如果各 rank 先求部分和再跨 rank 相加，加法的结合顺序变了，就不逐位相同了（官方参照实现自己的多 rank 模式就是后一种）。
- **torch 的对称内存**默认把设备号相同的 rank 当成在同一张卡上而拒绝（`detected allocations from overlapping devices`）。每个 rank 只看得见自己的卡时都是 0 号，要设 `TORCH_SYMM_MEM_ALLOW_OVERLAPPING_DEVICES=1`。DeepInfer 在 `deepinfer/__init__.py` 里设了。
- **`mega_mhc`、`bf16_mega_gate` 按 stream 各有一小块缓冲**，第一次在某个 stream 上调用时建，在录制中建会断言失败。引擎的录制器先在录制 stream 上预热，没问题；自己写测试录 graph 时要先在那个 stream 上跑一遍。
- **`SymmBuffer` 的布局取决于有没有共享专家**：建的时候带了共享专家，调用时就必须带共享专家的权重，否则 kernel 按另一种布局读缓冲区。所以空闲的 rank 调的也是带共享专家的那个。
- **`rsinfer serve --cuda-graph` 默认是 off**，默认的桶到 8 行为止。量吞吐要给 `--cuda-graph auto --graph-buckets 1 2 4 8 16 32 64`。
- **两个 worker 共用一张 24 GB 的卡**时，`--mem-fraction` 要调到 0.4 左右：部署工具已经把份额按卡上的 worker 数平分，但参照 kernel 的中间结果还要地方。
- **真权重前 8 层放一张卡**要 `--mem-fraction 0.9`：0.85 放不下状态，0.95 不给 prefill 留够它一次要的 12 GiB（`tilelang` 的 prefill 里有一个 12 GiB 的中间量，没查是哪个 kernel）。
- **`core.manifest` 不许单元模块里有可变的全局量**：`dict` 要写成 `MappingProxyType`，进程级的对象放在单元实例上。
- **服务是经代理拿到的**：一个"列表"服务不要在初始化时拷成 list，否则提供方重载后拿的是旧对象而不报错；每次经代理下标访问。
- DeepGEMM 在这台节点上打印 `MLOPart is unavailable; localization has been disabled`（它的一个探测子进程断言失败），不影响结果。

## SM100 上的任务

| 任务 | 内容 | 结果 |
|---|---|---|
| `20261115T120455Z-a5332691` | DeepInfer 自己，EP4/DP4 | 文本正常；decode 每卡 3378–3402 tokens/s，prefill 每卡 43–45k |
| `20261115T124601Z-6808bb23` | 真实形状 4 卡对 1 卡；整个模型 `tilelang` | 文本相同；整个模型第一次出有意义的文本 |
| `20261115T132755Z-1c6c0927` | 真权重 8 层 4 卡对 1 卡；整个模型开 EPLB 对不开 | 都相同；重排 15 次，1.344 → 1.050 |
| `20261115T134749Z-9f886b7d` | 融合的一对对不融合的；单卡真权重 8 层 | 差 7e-4 / 3.5e-3；`tilelang` 16 步逐位相同；`deepinfer` 带 graph 3.0 ms 一步 |
| `20261115T140515Z-63dd3d54` | kernel 测试（含融合的一对录 graph 回放） | 91 个全过 |
| `20261115T141007Z-f47417ae` | 整个模型 4 卡 `deepinfer` 带 graph；吞吐 | 4 个 worker 都录了图；129 tokens/s（4 并发）；256 并发 2396 tokens/s，每 token 32.8 ms |

吞吐只跑了一遍、没有预热，64 并发那一档里有编译；数的含义和怎么和 DeepInfer 比见 specs 的 10.7 节。

## 没做完的（交接）

见 [../specs/v1-ds1017.md](../specs/v1-ds1017.md) 10.8 节。最要紧的三件：

1. attention 那半边的融合（FlashMLA 的融合 kernel；`mega_mhc` 直接给 FP8 输入），这是和 DeepInfer 剩下的主要差距。
2. prefill 的预热：冷的编译缓存下，第一次 prefill 中途编译可能让别的 rank 在 `mega_moe` 里等过 60 秒。
3. 上一节「怎么做的」里列的三种热改，在多 rank 引擎上还没试过。
