# GLM-5.3-Flash 原生 RSI 后续迭代

分支 `dev/ascend_a3_gg`，权重 `/local/models/GLM-5.3-Flash-w8a8`。同一台 Ascend 910C 使用 16 个 die，DP2 × TP8、每域 EP8、DFC 和 `--acl-graph required`。除 MTP5 对照外均使用 MTP4。外部 MemCache 池 PID 为 `2268032`。

`sync_epochs()` 开发期间曾重启一次（`867813b5e15b2ff5`）。`iter2b` 后为恢复 ACL stream 再次重启（`8dadd59d3815274b`）。其他迭代在运行中的引擎执行 `control apply --wait-empty 60`，16 个 worker PID 保持，`restart_engine=False`。进入评分的配置均通过原脚本 6/6 精度，使用 `verify-aisbench-parity.py` 与 `cmp` 核对 `glm-8192-32` 两组输入正文。 [原始 CSV](../../../ascend-a3-results/2026-11-26/acl-graph/rsi-iter/ledger.csv)

## 逐轮结果

| 迭代 | 决定 | 身份 | 执行 / 暂停（秒） | 精度 | 16K 冷组 tok/s | 16K TTFT 均值 / P90（ms） | 16K TPOT 均值 / P90（ms） | 60K 暖组 tok/s | 60K TTFT 均值 / P90（ms） | 60K TPOT 均值 / P90（ms） | 配置与说明 |
| --- | --- | --- | --- | :---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `vllm_ref` | 参考 | `vllm-0.30.0rc2` | — | 6/6 | 79.9364 | 2588.3 / 3278.4 | 29.4 / 34.4 | 110.3326 | 1348.3 / 2325.5 | 23.4 / 27.4 | `glm-8192-32`，DP2×TP8、EP16、MTP5、INT8 KV |
| `native_head` | 原生对照 | `exact-restored-final` | — | 6/6 | 77.6184 | 2716.0 / 3435.4 | 27.6 / 35.4 | 113.5635 | 1322.9 / 1895.9 | 22.5 / 27.0 | DP2×TP8、EP8、DFC、SP、DSA、共享专家分片、融合 reduce-scatter、MTP4 |
| `iter1_indep_domains` | **保留** | `586235ec88ef36f6` | 2.655 / 0.021 | 6/6 | 100.4198 | 2215.9 / 3216.4 | 22.1 / 30.3 | 144.9451 | 1143.1 / 1578.8 | 17.7 / 23.2 | `sched.independent_domains=true`、`prefill_interval=0`、`balance_prefill=true`、`coalesce=0.008` |
| `iter2a_kda_mla_fused` | 拆分对照 | `867813b5e15b2ff5` | — | 6/6 | 98.5275 | 2076.4 / 3224.0 | 23.4 / 39.5 | 135.9952 | 1050.5 / 1590.7 | 19.2 / 25.7 | `kda_fused_fg_bmm` + `mla_fused_proj` 改善 TTFT，decode TPOT 回落 |
| `iter2a_mla_only` | **保留** | `846969c64e1c076a` | 7.722 / 0.029 | 6/6 | 98.5665 | 2159.9 / 3214.1 | 22.8 / 27.4 | 144.7676 | 1182.1 / 1567.1 | 17.1 / 20.1 | `mla_fused_proj=true`、`kda_fused_fg_bmm=false` |
| `iter2b_ag_mm` | 禁用 | `655a80b23613c028` | 6.692 / 0.022 | 0/6 | — | — | — | — | — | — | `npu_all_gather_base_mm` 在域 0 的第 1539 步报 HCCL `EI0019`，端口 `60000` / `60001` 绑定失败 |
| `iter2c_mla_tbmm` | 回退 | `9625ee816d9e613f` | 7.175 / 0.021 | 6/6 | 100.0390 | 2122.5 / 3222.0 | 22.9 / 31.3 | 141.4987 | 1167.9 / 1741.8 | 17.3 / 22.2 | `npu_transpose_batchmatmul` 使 60K 吞吐从 144.77 降到 141.50 tok/s，P90 时延上升 |
| `iter3a_draft_opt` | **保留** | `540e6ef419679629` | 7.375 / 0.021 | 6/6 | 102.5604 | 2228.4 / 3204.1 | 21.2 / 29.9 | 145.3082 | 1136.6 / 1551.4 | 16.8 / 20.8 | `glm53.fused_eh_norm=true` + `glm53.local_draft_argmax=true` |
| `iter3b_mtp_share_index` | **保留** | `84df25d57f963c31` | 7.130 / 0.021 | 6/6 | 102.6646 | 2294.3 / 3187.1 | 20.0 / 26.7 | 150.2680 | 1176.2 / 1551.2 | 16.1 / 20.2 | `glm53.mtp_share_index=true` |
| `iter3c_mtp5` | 回退 | `375fdee590017d08` | 11.830 / 0.022 | 6/6 | 101.9518 | 2099.0 / 3129.1 | 21.6 / 26.8 | 150.8823 | 1069.1 / 1557.6 | 17.0 / 21.5 | MTP5 冷组吞吐与两组 TPOT 回落，恢复 MTP4 |
| `iter4a_npu_overlap` | 回退 | `982dcb5d05926a1c` | 7.058 / 0.020 | 6/6 | 95.9997 | 2171.4 / 3149.9 | 21.9 / 32.9 | 145.1578 | 1234.1 / 1882.0 | 15.6 / 17.9 | 60K TPOT 最低，为 15.6 / 17.9 ms，每次调用 `pin_memory()` 后整体吞吐下降 |
| `iter4b_npu_ring_overlap` | 回退 | `78eb6e04dc862d19` | 0.121 / 0.019 | 6/6 | 96.4928 | 2540.0 / 3211.1 | 19.7 / 37.1 | 148.3346 | 1169.3 / 1895.2 | 16.6 / 22.5 | D2H 环形 pinned 缓冲和 `npu.Event` 的吞吐仍低于 `iter3b` |
| **`final_verified`** | **保留** | `c573f7c95f782354` | 7.161 / 0.021 | 6/6 | **103.5465** | **2334.1 / 3149.4** | **19.6 / 27.1** | **153.7484** | **1066.1 / 1547.4** | **16.7 / 20.3** | 保留的选项全部启用，MTP4 |

最终一轮比前版原生 16K / 60K 吞吐提高 33.4% / 35.4%，比同机 vLLM Ascend 提高 29.5% / 39.3%。16K 平均 TPOT 从 27.6 降至 19.6 ms。60K 平均 TTFT 从 1322.9 降至 1066.1 ms，平均 TPOT 从 22.5 降至 16.7 ms。

## 保留的改动

- **调度域独立推进**。每个域独立发计划并按序提交结果，快域可以先回报。复制前缀前等待在途步骤，排空和 `quiesce()` 通过 `sync_epochs()` 对齐各域的 epoch，保持热更协议的暂停语义。 [调度单元](../../engine/core/sched/loop.py) · [排空单元](../../engine/core/sched/drain.py) · [域调度测试](../../tests/units/scheduler/test_domains.py)
- **MLA 与索引器输入投影融合**。分别合并 BF16 的 `q_a_proj` / `kv_a_proj_with_mqa` 和 FP32 的 `indexer.wk_weights` / `indexer.index_kpool_compress_gate`。每组合并为一次 GEMM。 [GLM 层](../../engine/layers/glm53/unit.py)
- **草稿输入融合与词表分片 argmax**。Triton `fused_eh_norm` 合并位置 0 的 embedding mask、`enorm`、`hnorm` 和拼接。每个 TP rank 只对自己的草稿词表分片做投影，跨 rank 汇总本地最大值及全局索引。 [GLM 模型](../../engine/models/glm53/unit.py) · [投机执行](../../engine/executors/pytorch/spec.py)
- **草稿步复用 DSA 索引**。首个草稿 decode graph 计算稀疏 top-k，写入该 bucket 的固定设备缓冲。后续草稿 graph 复用该缓冲，并继续更新 KPool 和尾部缓存。 [录图与回放](../../engine/core/plan/unit.py) · [GLM 层](../../engine/layers/glm53/unit.py)

## 回退的改动

- KDA 的 `f_b_proj` / `g_b_proj` 批量 BMM 使 16K / 60K 平均 TPOT 从 22.1 / 17.7 增至 23.4 / 19.2 ms。
- 在 prefill 使用 AIV `npu_all_gather_base_mm` 时，域 0 出现 HCCL `EI0019` 端口绑定失败，精度门禁未通过。
- MLA 使用 `npu_transpose_batchmatmul` 后，60K 吞吐从 144.77 降至 141.50 tok/s，P90 时延上升。
- MTP5 使 16K 吞吐从 102.66 降至 101.95 tok/s，两组平均 TPOT 同时上升。恢复 MTP4。
- 全量异步 H2D/D2H 与 D2H 环形缓冲两项实验的 16K / 60K 吞吐分别为 96.00 / 145.16 和 96.49 / 148.33 tok/s，均低于异步之前的 102.66 / 150.27 tok/s。

## 热更和复现参数

最终一轮在运行中的实例排空后提交，身份为 `c573f7c95f782354`，16 个 worker PID 保持。检查与计划 4.096 秒，prepare 0.711 秒，排空 0.009 秒，暂停 0.021 秒，execute 7.161 秒，commit 2.508 秒。 [最终评分与事务](../../../ascend-a3-results/2026-11-26/acl-graph/rsi-iter/evals/final_verified)

整机使用 `--glm-dp-size 2 --glm-moe-backend dfc --glm-prefix-peer --overlap --acl-graph required`，prefill 预算为 8192。与下列配置组合。 [原生 GLM 部署](../../deploy/ascend.md#rsinfer-原生-glm-53-flash) · [复现 AISBench 输入与精度验收](../2026-11-26-ascend-npu-workflow.md#glm-aisbench-对照)

```text
--set sched.independent_domains=true \
--set sched.balance_prefill=true \
--set sched.prefill_coalesce_s=0.008 \
--set sched.prefill_interval=0 \
--set sched.prefill_occupancy_target=0 \
--set glm53.sequence_prefill=true \
--set glm53.index_prefill=true \
--set glm53.shared_expert_dp=true \
--set glm53.prefill_matmul_reduce_scatter=true \
--set glm53.mla_fused_proj=true \
--set glm53.fused_eh_norm=true \
--set glm53.local_draft_argmax=true \
--set glm53.mtp_share_index=true \
--speculative-algorithm EAGLE --speculative-num-steps 4 --speculative-num-draft-tokens 5
```
