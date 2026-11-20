# 接入真 kernel 之前的 serving 基线：master 的引擎（2026-11-11，SM90-80GB，4 层）

测的是 `dev/gw/attn-kernel` 分支当时的服务，引擎固定为 `03579e36a81900d8506f43f75041cc328985db56`。这个提交相对 master `ffa7254` 只多了文档和一个微基准脚本，`engine/` 完全相同，所以**这就是 master（接入 FlashInfer 和 Triton GDN 之前）在 SM90-80GB 上的基线**。从 [PR #51](https://github.com/open-agent-infra/rsinfer/pull/51) 的 `2f3ac834ba3e009d0f394f8a600e075eb2bdde67` 复用 `tests/gpu/bench_serving.py` 的 HTTP 负载和计时逻辑。没有合入或运行该 PR 的 FlashInfer、Triton GDN 实现。

这份记录补充 [attention 算子微基准](attn-kernel-benchmark.md)：这里测的是完整 HTTP 服务链路，但模型只有 Qwen3.8-27B 结构的前 4 层，使用 seed 1 的随机 BF16 权重（约 7.57 GiB），不代表完整 27B 或真实权重的回答质量。

## 环境与来源

- 通过 ossrun 运行；NVIDIA SM90-80GB，SM90，物理 GPU 1，81559 MiB；PyTorch 2.11.0+cu130，CUDA runtime 13.0。
- 当前分支的原有 GPU 环境，未使用为 PR #51 另行准备的依赖目录。GDN 保持 reference；eager attention 为 reference，graph 内 attention 为 `torch_graph`。执行脚本检查实际 kernel selection 记录。
- tokenizer 为 `Qwen/Qwen3.6-27B` 的真实 tokenizer，revision `8f33b98937c6031bb49e713dee3e8e185aafd26f`。`tokenizer.json` SHA256：`5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42`。
- 源码 release：`20261111T095717Z_03579e3_d7974fd0`，发布时 clean；源码包 SHA256：`70e5db0708b7bb63e612ef467b6c3175aa9f9734d12f4548657f96cadc086633`。
- PR #51 原始 benchmark SHA256：`bd736ebe82546c3beca0d1232ea3a2dc10fbe6da845004210c049b41856f0faf`。
- 本轮 orchestration 脚本 SHA256：`c1e24353f54ec185c767b47403002a2c08243164626ca16cdea436a9c422de15`；脚本作为 ossrun command payload 发送，没有改动引擎源码。

## 正确性检查

独立于性能计时，先在缓存的同一 checkpoint 上调用仓库已有的四项模型检查，再运行现有 HTTP 测试：

| 检查 | 范围 | 结果 |
|---|---|---|
| FP32 worker 对 HF | eager / graph，3 组输入，各生成 12 步 | token 全部相同；prefill logits 满足 `atol=rtol=1e-3` |
| BF16 worker 对 HF | eager / graph，3 组输入 | prefill logits cosine 为 0.99971–0.99981，最大绝对差 0.1484；通过已有 `cosine > 0.99` 阈值 |
| FP32 HTTP 对 HF | 真实 tokenizer，overlap 关 / 开，英文、代码、中文，各 8 步 | `tests/gpu/test_qwen38_serve.py`：2 passed，55.95 秒 |

BF16 的 greedy token 与 HF 开始分歧的位置为第 0、2、2 步（eager 和 graph 的上述记录相同）。这沿用当前项目的 BF16 验收口径，不能写成“BF16 token 全部与 HF 一致”，也没有证明任意长度 decode 的 logits 都满足同一数值界限。

验证 command：`cmd_20261111T095949Z_serving-baseline-validation_0aad59ea`，已取得 `done / exit 0` 和 `RSINFER_SERVING_BASELINE_VALIDATION_OK`。

## 性能测量口径

两次启动分别使用 `--cuda-graph off` 和 `--cuda-graph auto`。每个服务保持运行，通过控制面 `apply` 切换 `sched.overlap`，在没有请求时执行 dry-run 和 apply，并检查没有进程重启。其余参数固定：BF16、KV page size 16、`max_rows=64`、`max_context=4096`、`token_budget=2048`、`mem_fraction=0.85`、graph 默认桶 1/2/4/8、默认预算 1 GiB。没有使用 PR #51 报告中扩大 graph 桶、prefill 预算和调低显存比例的配置。

`auto` 下只覆盖到 8 行的图（master 的默认；#51 之后用 paged attention 时默认到 64 行）。并发 16/32 时，不能放进这些桶的步走 eager，因此这里叫 graph auto，不宣称高并发全程都在 replay。

- decode：32 个词的 prompt，生成 256 token；并发 1/4/8/16/32，每客户端 4 个计时请求。
- prefill：2000 个词的 prompt，生成 8 token；并发 1/4/8，每客户端 2 个计时请求。词数不当作实际输入 token 数，实际数量读取 usage。
- 每档每个客户端先发 1 个不计入统计的 warm-up 请求；客户端随后串行发送自己的请求，客户端之间并发。
- 每配置测量 2 次，overlap 顺序为关→开、开→关。相同配置间使用相同 seed 和 prompt，decode 与 prefill 使用不同 seed 区间。当前服务没有 prefix cache。
- 吞吐是整档总实际输出/输入 token 数除以整档 wall time；TTFT 是第一个非空文本片段的到达时间；TPOT 是首尾非空文本片段之间的时间除以实际输出 token 数减一。该指标包含 detokenizer 和流式分块行为，不能当成纯 GPU token 时间。这是 `2f3ac83` 那一版脚本的口径；#51 后来把它改成按每个 token 的事件计时（不管带不带文本）。
- 原脚本的 p99 只是本档少量请求的经验分位数；没有把它当作经过充分采样的服务 SLO。

原脚本遇到缺失 usage 时会采用请求中的 `max_tokens`。本轮在其 `one` 函数内添加严格检查（改过的脚本没有提交，在本地的 `.ossrun-artifacts/` 里），保留原计时位置和负载：每个请求必须收到 `[DONE]`、`finish_reason=length`、明确 usage，实际输出 token 数必须等于请求数，输入数必须为正、total 必须等于输入与输出之和；缺失信息直接失败。没有按 SSE chunk 数猜 token 数。

另外，对每个服务模式和 overlap 设置分别发送英文、代码、中文输入，检查流式拼接文本等于非流式文本、再次请求结果不变，并用 tokenizer 独立核对输入 token 计数。没有把不同 batch 或不同 kernel 的 BF16 文本一致作为性能测量的前提。

## 结果

全部 **64/64 次测量**完成：32 个配置各测 2 次；2160 个计时请求共生成 501376 个 token。含预热共 **2752 个请求**通过严格 SSE/usage 检查；另有 **12 组**流式/非流式/重复请求一致性检查通过（每组 3 个请求）。初末 GPU 采样均无其它运行进程，结束后物理 GPU 1 回到 4 MiB。

### Decode

下表是两次整档吞吐的中位数，单位为**输出 token/s**，没有把所有请求混在一起重算分位数。

| 并发 | eager | eager + overlap | graph auto | graph auto + overlap |
|---:|---:|---:|---:|---:|
| 1 | 139.4 | 166.1 | 253.0 | 353.8 |
| 4 | 499.3 | 586.7 | 777.4 | 1142.5 |
| 8 | 917.9 | 1067.7 | 1394.5 | 1810.6 |
| 16 | 1554.5 | 1777.1 | 1565.4 | 1819.5 |
| 32 | 2233.8 | 2450.7 | 2227.0 | 2482.7 |

对于 graph auto + overlap，TTFT/TPOT/延迟如下；每列取两次运行各自请求中位数的中位数。

| 并发 | TTFT（ms） | TPOT（ms） | 请求延迟（ms） |
|---:|---:|---:|---:|
| 1 | 30.9 | 2.71 | 723.6 |
| 4 | 69.6 | 3.24 | 895.9 |
| 8 | 122.2 | 3.94 | 1130.3 |
| 16 | 236.1 | 7.87 | 2243.8 |
| 32 | 399.1 | 11.34 | 3295.5 |

相对默认 eager、不重叠，graph auto + overlap 在并发 1/8/32 分别为 **2.54× / 1.97× / 1.11×**。8→16 并发时吞吐几乎不涨，TPOT 从 3.94 ms 升至 7.87 ms；这与默认 graph 桶只到 8、更多行的步走 eager 的行为相符。这里没有 profile 到每一步的执行分布，不能把全部差异归因于桶覆盖。

### Prefill

这是 2000 输入 token、8 输出 token 的整请求吞吐，包含短 decode，不是纯 prefill kernel 时间。各次实际 `prompt_tokens_mean` 均为 2000。单位为**输入 token/s**。

| 并发 | eager | eager + overlap | graph auto | graph auto + overlap |
|---:|---:|---:|---:|---:|
| 1 | 17965 | 19558 | 22521 | 24687 |
| 4 | 25543 | 27839 | 28083 | 30536 |
| 8 | 27993 | 28699 | 29723 | 30841 |

graph auto + overlap 的 TTFT 中位数在并发 1/4/8 分别为 **60.8 / 210.4 / 344.1 ms**。吞吐更高不意味着每档 TTFT 都更低：例如并发 8 的默认 eager TTFT 为 296.1 ms。

原始 JSON 保存每次运行的 median 和经验 p99。只有两次重复及少量请求，不作显著性检验或稳定尾延迟承诺。以上均为接入之前的基线，不能和另一张卡上的 PR #51 数字直接相除归因于 kernel；同一张卡、同一份源码上的配对比较见 [serving-flashinfer-gdn-ablation.md](serving-flashinfer-gdn-ablation.md)。

## 复现与原始记录

引擎和 benchmark 来自不同的固定提交。复用脚本时先取 PR #51 的该文件，不能把整个 PR 的实现混进基线：

```sh
git show 2f3ac834ba3e009d0f394f8a600e075eb2bdde67:tests/gpu/bench_serving.py > /tmp/bench_serving-pr51.py
# 在引擎 commit 03579e3 的环境中启动；MODEL 为同一个 4 层 checkpoint。
uv run python deploy/qwen38.py serve --model "$MODEL" --tokenizer "$TOKENIZER" \
  --dtype bf16 --conv-dtype bf16 --cuda-graph off --max-rows 64 --max-context 4096 \
  --token-budget 2048 --mem-fraction 0.85
uv run python /tmp/bench_serving-pr51.py --url http://127.0.0.1:8000 --name baseline \
  --concurrency 1 4 8 16 32 --requests 4 --prompt-words 32 --max-tokens 256 --output decode.json
uv run python /tmp/bench_serving-pr51.py --url http://127.0.0.1:8000 --name baseline \
  --concurrency 1 4 8 --requests 2 --prompt-words 2000 --max-tokens 8 --seed 100000000 --output prefill.json
```

以上为原始脚本的手动复现；本轮严格校验、热改 overlap、重复测量和环境记录的完整命令保存在 ossrun command payload 及本地 `.ossrun-artifacts/serving-baseline-benchmark-command.sh`。该命令 SHA256 为 `7b245db9fd40488bf61cfb2952a6e60c2bd711bd637de0a67f54c1782618d4af`。

性能 command：`cmd_20261111T100619Z_serving-baseline-benchmark_04cf7b97`。本地 `.ossrun-artifacts/` 保存验证日志、性能日志、原 benchmark、orchestration 脚本、tokenizer manifest 和 `serving-baseline-provenance.json`。生成物不提交。

本地（macOS）当时 `uv run pyright` 为 0 错误；全量 `uv run pytest -n 8 -x` 没有跑完（116 passed、5 failed 后首错即停，堆栈里是只读快照和进程组清理的 PermissionError）。这是本机的限制，和上面的远程 GPU 结果无关。

性能任务的最终标记已写出，末次 GPU 采样时间为 2026-11-11 10:19:57 UTC；原 command 的完成状态回写滞后，未把 `running` 当作成功证据。通过独立只读回收任务 `cmd_20261111T101951Z_probe_serving-baseline-fetch_b3745bdd`（`done / exit 0`、`RSINFER_BASELINE_ARTIFACT_FETCH_OK`）取得完整结果，核对 `complete=true`、64 个唯一测量组合、12 项协议检查、2752 个审计请求、引擎/benchmark/orchestration SHA256 和原任务最终成功标记。原始回收日志为 `.ossrun-artifacts/serving-baseline-fetch.log`，提取结果为 `serving-baseline-result.json`。

同一 checkpoint 的权重校验值（回收时计算，计算发生在压测结束后）：

| 文件 | 字节数 | SHA256 |
|---|---:|---|
| `model-00001-of-00002.safetensors` | 6618679872 | `4358450d4f4c5cd3df46b9790d60bb0f4c7687be71eddb4a554e9e3d34e28b33` |
| `model-00002-of-00002.safetensors` | 1511081224 | `66980f24c407fb11b390738df8dabe8869a8cd954419d8481e23aca8659ef7f5` |
