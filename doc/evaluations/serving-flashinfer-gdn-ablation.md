# FlashInfer 与 Triton GDN 的 serving 消融比较（2026-11-11，SM90-80GB）

上一轮 [serving 基线](serving-baseline-before-flashinfer.md) 只测了尚未接入 FlashInfer 的 `dev/gw/attn-kernel`。本轮实际运行 [PR #51](https://github.com/open-agent-infra/agxinfer/pull/51) 的 FlashInfer attention 和 Triton GDN 实现，拆开比较各自及组合后的 HTTP 服务性能。

## 比较边界

引擎和 `tests/gpu/bench_serving.py` 均固定在 PR #51 的 `2f3ac834ba3e009d0f394f8a600e075eb2bdde67`。在远程临时目录复制四份 `engine/`，只在复制时省略指定 kernel 单元的 `unit.toml`，四组统一增加下文的 padding 修复，其余源码逐字相同；记录省略清单、文件数和整棵树的 SHA256。所有组使用 `auto` 选择 backend，并核对实际解析记录。

| 组 | FlashInfer 单元 | Triton 单元 | eager attention | graph attention | GDN 与 gated RMSNorm |
|---|---|---|---|---|---|
| native | 不注册 | 不注册 | reference | torch_graph | reference |
| FlashInfer | 注册 | 不注册 | FlashInfer paged | FlashInfer paged | reference |
| Triton GDN | 不注册 | 注册 | reference | torch_graph | Triton |
| 两者 | 注册 | 注册 | FlashInfer paged | FlashInfer paged | Triton |

Triton 因子包含 `norm.rms_gated@1`，不只包含 recurrent update；Gemma RMSNorm 保持 reference。不能只设 `gdn.backend=reference` 就称为无 Triton，因为 gated RMSNorm 通过 catalog 独立选择。也不能强制 `attn.backend=reference` 代替本表的 native，因为那会排除 `torch_graph`，改变 graph 可用性。

本轮 native 是 PR #51 的配对对照，保留 native attention 路径，但包含该 PR 的 padding、metadata 和 graph 初始化变化，不与上一轮引擎 `03579e3` 字节相同。性能归因以本轮同源码、同参数的配对结果为准；没有将两个 PR 合并；FlashInfer/Triton 实现仍来自 PR #51 的隔离快照，设计分支只提交此次必要的 padding 修复、回归测试和评估记录。

## 固定环境与负载

- 通过 ossrun，在同一 SM90-80GB 物理 GPU 1 上串行运行；SM90，81559 MiB。PyTorch 2.11.0+cu130、CUDA runtime 13.0、FlashInfer 0.7.2、Triton 3.6.0、Transformers 5.22.1。
- 依赖环境按 PR #51 的锁文件独立创建：`794d8b37f9daf2ff0bf85e6db71aae250a02dfe6e38b19185b0cf18ae082468c`。local package mirror安装仍核验锁文件 SHA256；复用已校验的 CUDA 13.0 PyTorch wheel，不改原基线环境。
- 与上一轮相同的 Qwen3.8-27B 结构前 4 层（3 GDN、1 full attention），seed 1 随机 BF16 权重；相同真实 tokenizer。权重与 tokenizer 哈希见上一轮记录。这不是完整 27B 或语言质量评估。
- 每组分别测 `--cuda-graph off` / `auto`，固定开启 overlap。BF16、conv BF16、page size 16、max rows 64、max context 4096、token budget 2048、mem fraction 0.85、graph budget 1 GiB；graph 默认桶 1/2/4/8。并发 16/32 包含 eager fallback。
- decode：32 输入 token、256 输出 token，并发 1/4/8/16/32，每客户端 4 个计时请求。
- prefill：2000 输入 token、8 输出 token，并发 1/4/8，每客户端 2 个计时请求；整请求吞吐包含短 decode。
- 每档先预热每客户端 1 个请求，再计时；每组两次重复，相同 prompt 和 seed。TTFT 按第一个非空文本片段；TPOT 按首尾非空片段间隔除以实际输出 token 数减一。
- 复用原 benchmark 的计时逻辑，加入 `[DONE]`、finish reason 和显式 usage 的严格检查，不用请求的 `max_tokens` 填补缺失 usage。先做流式/非流式/重复请求一致性与 tokenizer 计数检查。

## 正确性与热更新

`tests/gpu/test_real_kernels.py`：**11 passed，17 warnings，77.00 秒**。涵盖 kernel 对 reference 的数值与状态比较、FlashInfer paged extend/decode、padding、不同 batch 的 CUDA Graph replay、kernel 单元重载，以及不支持的 FP32 显式 backend 拒绝启动。warnings 为第三方依赖的弃用提示。

修复后，四组 × eager/graph 共 **8/8** 项模型检查通过：3 组输入各生成 12 步，所有 decode logits 有限；每组自己的 eager 与 graph token 完全相同，prefill logits 在 `atol=rtol=2e-2` 内一致。四组 BF16 prefill 对 HF 均满足项目已有的 `cosine > 0.99` 阈值，但与 HF 的 greedy token 不完全相同。

| 组 | prefill cosine 范围（对 HF） | 最大绝对差 | 3 组输入与 HF 的 greedy 首次分歧位置 |
|---|---:|---:|---|
| native | 0.99971–0.99981 | 0.1484 | 0 / 2 / 2 |
| FlashInfer | 0.99972–0.99981 | 0.1562 | 0 / 4 / 2 |
| Triton GDN | 0.99966–0.99974 | 0.1602 | 9 / 2 / 2 |
| 两者 | 0.99966–0.99974 | 0.1562 | 1 / 2 / 2 |

分歧位置从 0 开始；这些是随机权重 BF16 数值检查，不是语言质量或逐 token 对 HF 等价的证明。

运行中的引擎按 native → FlashInfer → 两者 → Triton GDN 执行 `apply --source`；先 dry-run，再等待空引擎并提交。核对 `core.plan` 的 attention/GDN 实际选择与 1/2/4/8 桶，并保存 introspection。实际 kernel 单元的 ACTIVE 集合与四组一致，`core.plan` generation 按 1→2→3→4 递增；apply 仅 stage/remove kernel 和 reload plan，没有进程重启。四个阶段的 HTTP 协议检查通过。正式压测为每组分别启动服务，避免前一组保留的 workspace 影响资源比较；相同 graph auto 配置还核对热切换和冷启动的输出（FlashInfer、Triton GDN、两者三组，均 graph auto + overlap）。

### 测量前发现并修复的 padding 数值问题

未修改的 PR #51 在四组模型检查中，native 和仅 FlashInfer 的 eager/graph 对照通过；仅 Triton GDN 的 graph 生成则在第 1–3 个 token 后连续输出 token 0，与同配置 eager 不同。因此首次测量任务在正确性前置检查处终止，没有采纳该失败组合的吞吐。

独立 CPU 回归确认 `kernels.torch_graph` 的问题：长度外的 KV 地址替换为 slot 0，但只 mask attention score，没有清理对应的 V。若上游 padded lane 留下 NaN/Inf，`0 × NaN/Inf` 在加权求和时仍会污染有效行。Triton 的 padded lane 输出允许未定义；FlashInfer paged attention 不走这条固定长度求和路径。

最小修复在 gather 后对长度外的临时 V 填 0，保留原 score mask 和有效 token 的原始数值，不修改 cache、不对有效数据使用 `nan_to_num`。FP32/BF16 × NaN/+Inf/−Inf 六项回归在修复前全部失败；修复后 attention 单元测试 14 passed，整个 kernel 测试目录 30 passed。四组测量均使用该修复；不能将修复前的异常 graph 输出算成加速收益。

修复后的 `engine/kernels/torch_graph/ops.py` SHA256：`3a6e7c7fa409d2bd4f2a91f5f973c0d91a501ad4a76a56e5e46af71ce67cb124`。这是 kernel 数值修复，没有修改框架本体。

## 性能结果

全部 **128/128 次测量**完成：4 组 kernel × 2 种 graph 配置 × 8 档负载 × 2 次重复。4320 个计时请求共生成 1002752 个 token；含预热共 **5504 个请求**通过严格 SSE/usage 检查。另有 **36 组**流式/非流式/重复请求检查通过（每组 3 个请求）。初末 GPU 采样均没有运行进程，结束后 GPU 1 回到 4 MiB。

以下吞吐取两次整档速率的中位数；延迟取两次运行各自请求中位数的中位数，不是将所有请求混合重算分位数。所有组都开启 overlap。

### Decode：eager + overlap

单位：**输出 token/s**。

| 并发 | native | FlashInfer | Triton GDN | 两者 | 两者 / native |
|---:|---:|---:|---:|---:|---:|
| 1 | 164.7 | 167.9 | 205.2 | 204.0 | 1.24× |
| 4 | 581.1 | 636.5 | 728.3 | 797.5 | 1.37× |
| 8 | 1044.8 | 1215.7 | 1350.5 | 1540.4 | 1.47× |
| 16 | 1734.9 | 2253.0 | 2269.1 | 2918.4 | 1.68× |
| 32 | 2417.3 | 3556.8 | 3438.0 | 5439.0 | 2.25× |

### Decode：graph auto + overlap

单位：**输出 token/s**。

| 并发 | native | FlashInfer | Triton GDN | 两者 | 两者 / native |
|---:|---:|---:|---:|---:|---:|
| 1 | 343.2 | 336.8 | 401.7 | 404.1 | 1.18× |
| 4 | 1105.0 | 1154.5 | 1417.1 | 1550.5 | 1.40× |
| 8 | 1733.6 | 1910.0 | 2483.1 | 2945.4 | 1.70× |
| 16 | 1728.0 | 2101.9 | 2322.9 | 2835.7 | 1.64× |
| 32 | 2421.4 | 3464.2 | 3567.8 | 5307.1 | 2.19× |

### Prefill：2000 输入 + 8 输出

单位：**输入 token/s**；这是包含短 decode 的整请求吞吐。

| 执行模式（均 overlap） | 并发 | native | FlashInfer | Triton GDN | 两者 |
|---|---:|---:|---:|---:|---:|
| eager | 1 | 19651 | 20554 | 33519 | 34745 |
| eager | 4 | 26820 | 28981 | 69055 | 67817 |
| eager | 8 | 28714 | 30082 | 86002 | 91173 |
| graph auto | 1 | 24796 | 24189 | 45391 | 44797 |
| graph auto | 4 | 29875 | 29666 | 83399 | 83344 |
| graph auto | 8 | 30431 | 30104 | 96303 | 98152 |

### Decode 延迟：graph auto + overlap

单位：ms。

| 并发 | 组 | TTFT | TPOT | 请求延迟 |
|---:|---|---:|---:|---:|
| 8 | native | 126.9 | 4.11 | 1179.0 |
| 8 | FlashInfer | 129.6 | 3.68 | 1068.4 |
| 8 | Triton GDN | 25.8 | 3.11 | 821.0 |
| 8 | 两者 | 25.3 | 2.61 | 696.1 |
| 32 | native | 439.8 | 11.47 | 3377.7 |
| 32 | FlashInfer | 460.4 | 7.38 | 2351.9 |
| 32 | Triton GDN | 52.1 | 8.71 | 2284.9 |
| 32 | 两者 | 59.0 | 5.76 | 1537.5 |

### 结论与局限

并发 32 decode：eager 下仅 FlashInfer、仅 Triton GDN、两者分别是 native 的 **1.47× / 1.42× / 2.25×**；graph auto 下分别为 **1.43× / 1.47× / 2.19×**。这次测到了两项优化各自及组合的端到端收益，而不是用单层 attention 倍数推算 serving。

并发 8 prefill、graph auto：仅 FlashInfer 为 native 的 **0.99×**，仅 Triton GDN 为 **3.16×**，两者为 **3.23×**；TTFT 从 native 的 **348.9 ms** 降至两者的 **89.6 ms**。本配置下长 prompt 的主要收益来自 GDN 优化，单加 FlashInfer 没有显示出收益。该模型的前 4 层由 3 个 GDN 和 1 个 full-attention 层组成。

单请求 decode 的 FlashInfer 收益很小：eager 为 1.02×，graph auto 为 0.98×。两次测量不足以把这种小差异视为稳定改善或退化；native graph auto 的该档两次吞吐相差约 5.1%。全部测量中最大的两次速率差为 13.6%，出现在 eager、两者、prefill 并发 4。原始 JSON 保留每次速率、median 和经验 p99，不作稳定尾延迟或显著性承诺。

graph auto 默认只录到 8 行，16/32 并发包含 eager fallback，因此不保证比 eager 更快。这里只改变 kernel 组合，没有扩大桶、增加 prefill token budget 或改显存比例；没有完整模型、真实权重质量、SM100 或原生 SGLang 的配对比较。

## 复现口径

先检出 PR #51 的上述固定提交，再应用 `5e14f65` 中对 `engine/kernels/torch_graph/ops.py` 的两行修复。四份临时源码只省略下列 manifest，保留所有 Python 实现：native 省略 `kernels/flashinfer/unit.toml` 和 `kernels/triton/unit.toml`；FlashInfer 只省略 Triton；Triton GDN 只省略 FlashInfer；两者都启用时不省略。路径相对于 `engine/`。所有份使用同一修复文件，不能用未修复的 native 对照。

在同一 GPU 环境中，每份源码分别运行以下两种 `MODE`（`off` / `auto`）。`SOURCE` 为该份源码的父目录，`MODEL` 与 `TOKENIZER` 指向前述同一份缓存；不要把完整模型、另一组权重或其他 tokenizer 混进比较。

```sh
uv run python deploy/qwen38.py serve --source "$SOURCE" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --dtype bf16 --conv-dtype bf16 \
  --cuda-graph "$MODE" --overlap --max-rows 64 --max-context 4096 \
  --token-budget 2048 --mem-fraction 0.85
uv run python tests/gpu/bench_serving.py --url http://127.0.0.1:8000 --name "$NAME" \
  --concurrency 1 4 8 16 32 --requests 4 --prompt-words 32 --max-tokens 256 --output decode.json
uv run python tests/gpu/bench_serving.py --url http://127.0.0.1:8000 --name "$NAME" \
  --concurrency 1 4 8 --requests 2 --prompt-words 2000 --max-tokens 8 --seed 100000000 --output prefill.json
```

每组重复两次，保存不同输出文件；原脚本每档自带预热。以上为负载的手动复现。本轮增加的严格 SSE 校验、热切换、模型级检查、源码指纹与结果回收都在完整 ossrun command payload 中，文件名与哈希在下节记录。

## 可追溯记录

- 原始 kernel 验证 release：`20261111T095016Z_2f3ac83_f64c4097`，发布时 clean。
- 配对模型与 serving release：`20261111T110144Z_2f3ac83-dirty_912b719d`，以 PR #51 提交为底，仅增加上述 padding 修复；源码包 SHA256：`06fef5b2e71816ca096180e6320d8d7d07e4e799ef7222572201b146b325032b`。
- 依赖准备：`cmd_20261111T104859Z_serving-ablation-env_cfb32d47`，done / exit 0，`AGXINFER_ABLATION_ENV_OK`。
- kernel 验证：`cmd_20261111T105148Z_serving-ablation-kernel-validation_0d7d3cc2`，done / exit 0，`AGXINFER_ABLATION_KERNEL_TESTS_OK`。
- 首次模型检查：`cmd_20261111T105347Z_serving-ablation-model-validation_bf35a287`，失败记录保留。
- 修复后模型验证：`cmd_20261111T110254Z_serving-ablation-model-fixed_3f076cfb`，done / exit 0，`AGXINFER_ABLATION_MODEL_OK`。
- serving：`cmd_20261111T110308Z_serving-ablation-benchmark-fixed_7f9df458`；首次未修复的 command `cmd_20261111T105641Z_serving-ablation-benchmark_feb21891` 在正确性前置检查失败，没有开始计时。
- 原始日志、完整 command payload、模型结果和性能 JSON 保存在本地 `.ossrun-artifacts/` 与 ossrun command 记录；生成物不提交。
- 原性能 command 的 OSS 状态回写滞后，回收后仍显示 running，不能据此宣称原 command 已取得 exit 0。计算结束后的 GPU 采样为 2026-11-11 11:21:03 UTC。独立只读回收 command `cmd_20261111T112127Z_probe_serving-ablation-final-fetch_4250d75b` 为 done / exit 0；已取得 `complete=true`、原任务最终成功标记，核对 128 个唯一配置组合、36 项协议检查、5504 个请求以及引擎/benchmark/harness 哈希。回收日志为 `serving-ablation-final-fetch.log`，提取结果为 `serving-ablation-result.json` 与 `serving-ablation-model-result.json`。


| 本地 artifact / command payload | SHA256 |
|---|---|
| `serving-ablation-harness.py` | `0beca46fc447df3e3bc57f0962bc7ee8605b53be61134974314322125db75655` |
| `serving-ablation-model-check.py` | `0cea5f42055c89c62aeeb9899a7355a01a5699d963364c515c65aa3eafdf4983` |
| `serving-ablation-benchmark-fixed-command.sh` | `d99d5110bfc9c0bb587b897e17dd0c181c80b97a84d94aadf1b45dc0b49a77c6` |
| `serving-ablation-model-fixed-command.sh` | `29dfa30e3eef7c25bebdec1c9e3907943935b9edd79ef6754132e67c6d6ec0a7` |

## 本地提交验证

修复后 `uv run pyright`：0 errors / warnings。`uv run pytest tests/units/kernels -q`：30 passed。`uv run pytest -n 8 -x`：116 passed、5 failed 后提前停止（22.34 秒），没有完成全量；失败堆栈含 macOS 上的只读快照和进程组 PermissionError。未修改框架或测试以绕过失败，未触发 formal 范围。
