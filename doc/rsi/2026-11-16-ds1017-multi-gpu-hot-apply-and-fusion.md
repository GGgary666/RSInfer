# DS 1017 多卡热改实测、attention 与 Engram 融合算子及吞吐对比（2026-11-16）

- 记录人：Antigravity。
- 对应：[../specs/v1-ds1017.md](../specs/v1-ds1017.md) 第 10.6–10.9 节；PR #66（分支 `dev/ds1017-ep`）。承接 [2026-11-15-ds1017-several-gpus.md](2026-11-15-ds1017-several-gpus.md) 的交接清单。
- 机器：`dev-server`（`pyright` 与 CPU 单测）；SM100 节点 `n1`（`gpu-node-n1`，GPU `4,5,6,7` 跑 4 卡全量模型引擎，GPU `1,2` 跑单/双卡 kernel 与状态重建单测）。

## 结论

1. **Attention 半边融合与分布式 `EngramBuffer` 已全部接通并录进 4 卡 CUDA Graph**：
   - 在 4096 上下文、每卡 64 个并发请求（总计 256 路）的稳态 decode 对照下，默认 `sched.overlap=false` 达到 **3,360.2 tok/s/卡**（**19.05 ms/步**），与同节点 DeepInfer 基线（3,402 tok/s/卡、18.8 ms/步）基本持平（**98.8%**；交接时未融合 attention 半边为 32.8 ms/步、57%）；开启 `sched.overlap=true` 把 4 个调度域的 CPU 组批与 IPC 开销藏进上一步 GPU 回放后，稳态 decode 达到 **4,534.3 tok/s/卡**（总计 **18,137.2 tok/s**，**14.11 ms/步**，DeepInfer 单缓冲循环的 **133.3%**）。
2. **多卡在线热改（`control apply`）全链路跑通**：
   - 不碰权重的算子代码热改与集体缓冲重建约 **59–72 秒**（对比 `/shared-fs` 全量读盘冷启动约 18–20 分钟，快约 **16 倍**），纯配置/录图与开启 EPLB 约 **3.3–9.5 秒**；忙时默认拒绝、`--wait-empty`（新请求返 `503`、在途请求 `200` 排空后自动提交）、`--force` 均符合规范；需要重排专家栈形状的原地扩缩（`eplb.extra_experts: 0 → 32 → 0`，`384 → 416 → 384` 槽位）因需重读 `/shared-fs` 权重耗时约 **19–20.5 分钟**，全程无 OOM 干净 `committed`，所有贪婪续写与冷启动**逐字完全一致**。
3. **EPLB 接上 `deepinfer` + CUDA Graph 实测通过**：`extra_experts=0`（384 槽）与 `extra_experts=32`（416 槽，每卡 104 槽）在线重排均正常，带 32 个冗余槽位时最忙 rank 负载比从 `1.697` 压到 **`1.035`**（DeepInfer 同配置报告值为 `1.074`），文本逐字不变。
4. **多卡热改实测抓到并修掉 4 项真实问题**（全部在单元侧完成，零框架本体改动，CI 全绿）。

## 做了什么

| 部分 | 内容 |
|---|---|
| Attention 组合 definition | 新增 `ds.attn_qkv@1`、`ds.cache_attn_out@1`、`ds.sparse_attn_out@1`（`kernels/ds1017_compose.py`），参照与 `tilelang` 走原有算子组合，`deepinfer`（`kernels/deepinfer/mega.py`）走 `mega_mhc` 直接产出的 FP8 输入、`fp8_gemm_nt` 合并 `wq_a + wkv`、FlashMLA `fused_norm_rope_attn_rope_cast` 与分组/合并输出投影 |
| 分布式 Engram（`deep_ep.EngramBuffer`） | 新增 `ds.engram@1` 并为 `ds.engram_fetch@1` 增加 `stage` 模式：`deepinfer` 后端在多卡且装有 `deep_ep` 时建立 `EngramBuffer`（可录制进 CUDA Graph），在 `Layers.begin()` 发起异步 `stage_fetch`，与第 0 层计算重叠，在首个 Engram 层 `get_fetched` 后走 `fp8_fp4_gemm_nt` + `engram_gate_fwd` |
| TileLang 宽度对齐与 prefill 预热 | `kernels/deepinfer/ops.py` 将分页与非分页 indexer 的 top-k 宽度固定补齐到 `INDEX_PAD = 512`；`Layers._warm_prefill()` 在 worker 就绪时用小形状把单卡 prefill 算子提前编译一遍，避免多卡运行中途单个 rank 编译超时拖死 `mega_moe` |
| 生命周期与状态扩缩清理 | `Mega.close()` 显式销毁 `SymmBuffer` 与 `EngramBuffer` 并 `torch.cuda.empty_cache()`；`Mega._cached` 用 `weakref.finalize` + `_version` 跟踪缓存权重视图；`Mega.symm` 在专家物理槽位数变化时自动重建对称内存缓冲；`models.ds1017.close()` 卸载释放缓存与 `layers.ds1017` 回调 |

## 4×SM100 在线热改（`control apply`）耗时与行为矩阵

引擎配置：4 张 SM100（每卡 178.35 GiB），全量 40 层真权重，`--ds-backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 64 --disable-radix-cache`。

| 场景 | 命令 / 变更 | 结果 | 墙钟耗时 | 说明 |
|---|---|---|---:|---|
| 代码热更新上融合算子 | `apply --source ep-work`（同时启用 attention 融合 + `EngramBuffer` + `_warm_prefill`） | `committed` | **72.02 s** | 不重读权重；热改后中英文贪婪续写与基线、以及随后冷启动结果**100% 逐字相同** |
| 双集体缓冲激活下重载 `kernels.deepinfer` | 连续两次改 `mega.py` 再 `apply --source` | `committed` | **59.43 s / 61.10 s** | 4 个 rank 同步销毁重建 `SymmBuffer` + `EngramBuffer` 并重录 `[1..64]` 全部 decode CUDA Graph，文本逐字一致 |
| 忙时默认拒绝 | 600 token 请求在途时直接 `apply --config` | `aborted` | **2.37 s** | 干净拒绝变更，在途请求正常 `HTTP 200` 完成 |
| 忙时 `--wait-empty` | 在途请求运行中 `apply --wait-empty 120` | `committed` | **9.53 s** | 排水期间新进请求收到 **`HTTP 503`**，在途 600 token 请求正常 **`HTTP 200`** 跑完后自动提交；恢复桶 `[1..64]` 耗时 **3.33 s** |
| 忙时 `--force` | 在途请求运行中 `apply --force` | `committed` | **6.50 s** | 立即中断在途请求并完成重录图提交，随后新请求 `HTTP 200` 且文本逐字一致 |
| 在线开启 `sched.token_budget=4096` / `sched.overlap=true` | `apply --config`（触发 `kernels.deepinfer` 按新预算扩建缓冲并重录图） | `committed` | **~60 s / ~4 s** | 用于 4096 上下文压测，压测后原地恢复默认配置 `committed` |
| EPLB 在线开启（无冗余槽，384 槽） | `apply --config '{"eplb.interval": 16, "eplb.window": 16, "eplb.extra_experts": 0}'` | `committed` | **4.23 s** | 8 路并发 `8/8 HTTP 200`，触发 3 次重排（最忙 rank 负载比 `1.890 → 1.020`），中文贪婪续写逐字一致 |
| EPLB 热扩冗余槽（`384 → 416` 槽，每卡 96→104） | `apply --config '{"eplb.interval": 16, "eplb.window": 16, "eplb.extra_experts": 32}'` | `committed` | **1228.11 s** | 先缩状态池、释放旧专家栈再从 `/shared-fs` 装载 104 槽位权重并重录图（`execute=1225.7s`）；`8/8 HTTP 200`，3 次重排（step 4: `1.697 → 1.078`，step 20: `1.386 → 1.036`，step 36: `1.216 → 1.035`），文本逐字一致 |
| EPLB 热缩恢复（`416 → 384` 槽，关闭 EPLB） | `apply --config '{"eplb.interval": 0, "eplb.window": 100, "eplb.extra_experts": 0}'` | `committed` | **1150.87 s** | 恢复 96 槽位权重与原状态池大小，显存回到 `{164580..164588} MiB`，英文贪婪续写逐字一致 |

## 吞吐对比（4×SM100，预热后测量）

### 1. 融合前后 HTTP 端到端吞吐（`tests/gpu/bench_serving.py`，约 201 输入 token、128 输出 token，默认 `sched.overlap=false`）

| 并发（每卡） | 融合前输出 tok/s | 融合后输出 tok/s | 融合前中位 TPOT | 融合后中位 TPOT | 融合前中位 TTFT | 融合后中位 TTFT |
|---|---:|---:|---:|---:|---:|---:|
| 4（1） | 304.3 | **359.6**（+18.2%） | 12.42 ms | **10.57 ms** | 0.104 s | **0.080 s** |
| 64（16） | 302.0 | **452.6**（+49.9%） | 32.22 ms | **29.43 ms** | 6.850 s | **3.132 s** |
| 256（64） | 1,849.4 | **1,852.4** | 47.57 ms | 70.95 ms | 9.722 s | **6.934 s** |

- 交接时 `c=64` 未预热测得 `183 tok/s`、p99 TTFT `73 s`；补齐 `INDEX_PAD = 512`、`_warm_prefill()` 与融合算子后预热稳态提升到 **452.6 tok/s**，中位 TTFT 缩短一半以上。
- `c=256`（512 个请求分两波进每卡 64 行的池子）整段墙钟受第二波请求排队进 prefill 穿插影响；纯稳态 decode 对比见下表。

### 2. 对标 DeepInfer 的 4096 上下文基准（每卡 64 请求 = 256 并发，无前缀缓存，无投机）

| 指标 | DeepInfer 基线（`examples/benchmark.py`） | 本引擎（`sched.overlap=false`） | 本引擎（`sched.overlap=true`） |
|---|---:|---:|---:|
| 稳态 decode 吞吐（每卡） | 3,402.0 tok/s | **3,360.2 tok/s**（98.8%） | **4,534.3 tok/s**（133.3%） |
| 稳态 decode 吞吐（4 卡总计） | 13,608.0 tok/s | **13,440.7 tok/s** | **18,137.2 tok/s** |
| 稳态 decode 步耗时（中位数） | 18.80 ms | **19.05 ms** | **14.11 ms** |
| 4096 上下文纯 prefill（每卡，经 HTTP） | 44,952.0 tok/s | **2,665.6 tok/s**（256 路混合阶段折算约 3,316 tok/s） | — |

- **Decode 结论**：接完 attention 半边融合与 `EngramBuffer` 后，GPU 上单步 64 行的图回放本身约 **14.1 ms**；不开 overlap 时加上 4 个调度域每步约 `4.9 ms` 的 Python 组批与 IPC 正好是 **19.05 ms**（追平 DeepInfer 的 `18.8 ms`）；打开 `--overlap` 把 CPU 组批完全重叠后直接释放出 **14.11 ms/步（4,534.3 tok/s/卡）**。
- **Prefill 差距原因**：当前 `sched.token_budget=4096` 下每卡每步只批 1 条 3801 token 的请求，且 prefill 走主机端构造 `Layout`、每层 `torch.cat` 拼装 key/index（尚未接入 DeepInfer 的 `kv_index_cat` 等设备端元数据算子）及 Python eager 逐算子发射，再加上经 HTTP + JSON + 分词器的端到端开销；后续若要追 prefill 峰值，需放大 `token_budget` 并把 prefill 元数据拼装搬上设备。

## 多卡热改与实测中抓到并修复的问题

1. **`_warm_prefill` 假索引触发单 rank `NaN` 与部分 rank 回滚卡死**：
   - 预热 `index_topk` 时 `cand = zeros` 让 `topk_select` 读了未初始化的列，仅 `worker/2` 踩中 `NaN` 抛异常；随后单 rank 单独回滚录带 `mega_moe` 的图时卡在 60 秒对称内存屏障。
   - 单元侧将预热候选改为 `[0, -1, ..., -1]` 修复；框架侧多卡部分 rank 失败回滚的集体操作死锁已记录在 [../framework-bugs/2026-11-16-partial-rank-reload-failure-hangs-rollback-collectives.md](../framework-bugs/2026-11-16-partial-rank-reload-failure-hangs-rollback-collectives.md)。
2. **`deep_ep 2.5.0` 在 HPC-X / RoCE 环境误报重复 NCCL**：
   - `worker.comm` 初始化 NCCL 后，进程映射里会出现网络插件 `libnccl-net.so.0.0.0`，`deep_ep` 导入时的子串检查 `libnccl` 把它误判为第二份 `libnccl.so` 而 `AssertionError`。在 `Mega._deep_ep()` 中按实际 `libnccl.so*` 库名判断并安全放行，双卡 `test_ds1017_deepinfer.py` **93/93 通过**。
3. **EPLB 配合 `--cuda-graph auto` 首步崩溃**：
   - 录图预热在 `torch.inference_mode()` 内首次触发 `Placement.start()`，导致 `window` / `counts` / `to_physical` 成为 inference tensor，而每步结束 `PyTorchExecutor` 在 `inference_mode` 外调 `model.stepped()` 原地写它们时报 `RuntimeError`。在 `Placement.start()` 用 `with torch.inference_mode(False):` 包裹分配，并补 CPU 回归单测 `test_placement_started_inside_inference_mode_steps_and_rebalances_outside_it`。
4. **原地热扩 `eplb.extra_experts=32`（每卡每层 96→104 槽位）OOM 与重排通信缓冲尖峰**：
   - `WeightStore.declare` 替换同名不同形状权重时未同步弹出 `self.tensors` 与 `self.decls`，旧张量引用未归零；
   - `eplb.extra_experts` 变更不重载 `layers.ds1017`，旧 `models.ds1017` 挂在 `layers.ds1017.set_tables` 上的绑定方法钉住了 `old_model._cached`（约 45.7 GiB 旧专家栈）；
   - `models.ds1017.start()` 原先先扩权重后缩状态池；
   - `Placement._move()` 跨卡搬迁专家时单次 `Group.gather` 拼齐全部离开槽位产生数 GiB 临时缓冲尖峰。
   - 四处均已在单元侧修好，补了 GPU 单测 `test_state_rebuild.py`（**3/3 通过**）与 CPU 多块重排单测，并补充记录到 [../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md](../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md)。

## 补充（2026-11-16 午）：Codex 评审的修复，以及 prefill 重测

### EPLB 重排后热重载模型

Codex 在双 rank 引擎上复现：EPLB 重排以后热重载 `models.ds1017`，映射回到初始顺序而权重留在搬过的位置，输出全变。修复和验证见 [../framework-bugs/2026-11-16-eplb-model-reload-resets-maps-without-restoring-weights.md](../framework-bugs/2026-11-16-eplb-model-reload-resets-maps-without-restoring-weights.md)：专家位置随权重保存在 `experts.placed`，模型或 `state.expert_placement` 重载时按它重建映射。SM100 4 卡上重排 44,202 个槽位后，热重载模型（4.3 s）、关掉 EPLB（4.5 s）、恢复源码（3.7 s），三次文本都和重排后逐字相同。在线增加 EP 卡数约定整引擎重启。

### prefill 重测（同一个引擎，热改 `sched.token_budget`）

上午表里的 2,665.6 tok/s/卡是 `sched.token_budget = 4096`（每卡每步只放一条约 3,800 token 的请求）、而且测量窗里混着第一次遇到的 `M` 档位的 JIT 编译。这次热改成 DeepInfer benchmark 的 32,768（71 s，`kernels.deepinfer` 按新预算重建缓冲区、`core.plan` 重录图），64 个约 3,801 token 的请求经 HTTP 连问三轮（每轮新的提示词，形状相同）：

| 轮 | 用时 | 每卡 tok/s |
|---|---:|---:|
| 0（第一次遇到这些形状，有 JIT 编译） | 29.46 s | 2,064 |
| 1 | 5.31 s | **11,459** |
| 2 | 5.56 s | **10,941** |

- 预热以后每卡约 1.1 万 tok/s，是上午数的 4 倍多，仍是 DeepInfer（44,952，进程内只计 `llm_engine.step()`）的约四分之一。每卡一轮约 6 万 token、两步，约 2.6 s 一步；DeepInfer 的同样一步是 727 ms。
- 单卡单独量的融合算子（`M = 32,768`，40 层的 `hc_step`、`sparse_attn_out`、`hc_step + moe` 加起来 435 ms）说明大头不在这几个 kernel。还没逐项量的候选：每个压缩层都重新 `load_values` 反量化 latent（38 层；DeepInfer 只在 4 个 KV 源层装一次、后面的层复用），每层两次 `torch.cat` 拼 attention 的源，indexer、压缩器、Engram 的 prefill 路径，主机上的 `lay_out`（约 10 ms 一步），以及 HTTP 和分词。下一步在跑着的引擎上加 profiler 热改进去量。
- 量的结果和之后的稳态吞吐见 [2026-11-16-ds1017-prefill-overhead.md](2026-11-16-ds1017-prefill-overhead.md)：大头是主机上的 `lay_out`，上面列的几个候选都不是。
