# DS 1017 多卡热改实测、attention 与 Engram 融合算子及热改问题修复（2026-11-16）

- 记录人：Antigravity。
- 对应：[../specs/v1-ds1017.md](../specs/v1-ds1017.md) 第 10.6–10.9 节；PR #66（分支 `dev/ds1017-ep`）。承接 [2026-11-15-ds1017-several-gpus.md](2026-11-15-ds1017-several-gpus.md) 的交接清单。
- 机器：`dev-server`（`pyright` 与 CPU 单测）；SM100 节点 `n1`（`gpu-node-n1`，GPU `4,5,6,7` 跑 4 卡全量模型引擎，GPU `1,2` 跑单/双卡 kernel 与状态重建单测）。

## 已完成内容

### 1. Attention 半边融合、分布式 `EngramBuffer` 与 prefill 预热

| 部分 | 内容 |
|---|---|
| Attention 组合 definition | 新增 `ds.attn_qkv@1`、`ds.cache_attn_out@1`、`ds.sparse_attn_out@1`（`kernels/ds1017_compose.py`），参照与 `tilelang` 走原有算子组合，`deepinfer`（`kernels/deepinfer/mega.py`）走 `mega_mhc` 直接产出的 FP8 输入、`fp8_gemm_nt` 合并 `wq_a + wkv`、FlashMLA `fused_norm_rope_attn_rope_cast` 与分组/合并输出投影 |
| 分布式 Engram（`deep_ep.EngramBuffer`） | 新增 `ds.engram@1` 并为 `ds.engram_fetch@1` 增加 `stage` 模式：`deepinfer` 后端在多卡且装有 `deep_ep` 时建立 `EngramBuffer`（可录制进 CUDA Graph），在 `Layers.begin()` 发起异步 `stage_fetch`，与第 0 层重叠，在首个 Engram 层 `get_fetched` 后走 `fp8_fp4_gemm_nt` + `engram_gate_fwd` |
| TileLang 宽度对齐与 prefill 预热 | `kernels/deepinfer/ops.py` 将分页与非分页 indexer 的 top-k 宽度固定补齐到 `INDEX_PAD = 512`；`Layers._warm_prefill()` 在 worker 就绪时用小形状把单卡 prefill 算子提前编译一遍，避免多卡运行中途单个 rank 编译超时拖死 `mega_moe` |
| 生命周期清理 | `Mega.close()` 显式销毁 `SymmBuffer` 与 `EngramBuffer` 并 `torch.cuda.empty_cache()`；`Mega._cached` 用 `weakref.finalize` + `_version` 跟踪缓存权重视图；`Mega.symm` 在专家物理槽位数变化（如热改 `eplb.extra_experts`）时自动重建对称内存缓冲 |

### 2. 4×SM100 在线热改（`control apply`）实测结果

在 4 张 SM100（全量 40 层真权重，`--ds-backend deepinfer --cuda-graph auto --graph-buckets 1 2 4 8 16 32 64`）上直接通过 `control apply` 演练多卡热改：

| 场景 | 命令 / 变更 | 结果 | 墙钟耗时 | 说明 |
|---|---|---|---:|---|
| 融合前基线压测 | `bench_serving.py`（预热后，`c=4 / 64 / 256`） | 通过 | — | `c=4`: 304.3 tok/s（TPOT 12.42 ms）；`c=64`: 302.0 tok/s（TPOT 32.22 ms）；`c=256`: 1849.4 tok/s（TPOT 47.57 ms） |
| 代码热更新上融合算子 | `apply --source ep-work`（同时启用 attention 融合 + `EngramBuffer` + `_warm_prefill`） | `committed` | **72.02 s** | 对比 `/shared-fs` 冷缓存启动约 18 分钟；热改后中英文贪婪续写与基线、以及随后冷启动结果**100% 逐字相同** |
| 双集体缓冲激活下重载 `kernels.deepinfer` | 连续两次改 `mega.py` 再 `apply --source` | `committed` | **59.43 s / 61.10 s** | 4 个 rank 同步销毁重建 `SymmBuffer` + `EngramBuffer` 并重录 `[1..64]` 全部 decode CUDA Graph，文本逐字一致 |
| 忙时默认拒绝 | 600 token 请求在途时直接 `apply --config` | `aborted` | **2.37 s** | 干净拒绝变更，在途请求正常 `HTTP 200` 完成 |
| 忙时 `--wait-empty` | 在途请求运行中 `apply --wait-empty 120` | `committed` | **9.53 s** | 排水期间新进请求收到 **`HTTP 503`**，在途 600 token 请求正常 **`HTTP 200`** 跑完后自动提交；恢复桶 `[1..64]` 耗时 **3.33 s** |
| 忙时 `--force` | 在途请求运行中 `apply --force` | `committed` | **6.50 s** | 立即中断在途请求并完成重录图提交，随后新请求 `HTTP 200` 且文本逐字一致 |
| EPLB + `deepinfer` + CUDA Graph 在线开启 | `apply --config '{"eplb.interval": 16, "eplb.window": 16, "eplb.extra_experts": 0}'` | `committed` | **4.23 s** | 8 路并发请求 `8/8 HTTP 200`，在线重排（step 4 搬迁 15,163 槽位，最忙 rank 负载比 `1.705 → 1.222`；step 20 `1.890 → 1.023`；step 36 `1.259 → 1.020`），贪婪中文续写与未开 EPLB 时**逐字相同** |

### 3. 多卡热改与实测中抓到并修复的问题

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
   - 四处均已在单元侧修好（`WeightStore.declare` 清理 `self.tensors`/`self.decls`/`self.filled`、`models.ds1017.close()` 卸载释放、扩权重前先缩状态池、`Placement._move` 按 16 条分块 `gather`），补了 GPU 单测 `test_state_rebuild.py`（**3/3 通过**）与 CPU 多块重排单测，并补充记录到 [../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md](../framework-bugs/2026-11-13-a-weight-dtype-change-runs-out-of-memory-and-its-rollback-is-unconfirmed.md)。

## 待收尾项（已在 SM100 `n1` 后台挂跑）

`/tmp/run_final_pipeline.sh` 已在 `n1`（GPU `4,5,6,7`）后台运行，输出写往 `/tmp/final_steps.log` 与 `/tmp/final_results.json`，依次完成：
1. 融合后预热态 `bench_serving.py`（`c=4 / 64 / 256`）吞吐对比；
2. 对标 DeepInfer 的 4096 上下文纯 prefill 与每卡 64 请求稳态 decode 吞吐（`overlap=false` 与 `overlap=true`）；
3. `eplb.extra_experts=32`（`384 → 416 → 384` 槽位）在线热扩缩与重排实测。
明天直接读取 `/tmp/final_results.json` 填入最终数字并清理 `n1` 临时目录即可。
