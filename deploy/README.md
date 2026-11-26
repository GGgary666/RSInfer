# 单机部署与命令行

从仓库根目录运行 `python deploy/rsinfer.py`。原生 `serve` 按 checkpoint 的 `config.json` 选择模型配方，`--run` 指定本实例的运行目录。子命令和参数以 `python deploy/rsinfer.py --help`、`python deploy/rsinfer.py serve --help` 为准。

## Ascend 910C

三种部署方式的启动参数、首请求和停服命令集中在 [Ascend 单机部署](ascend.md)。环境预检、调优和验证按任务查 [Ascend / CANN 文档导航](../doc/README.md)。

### RSInfer 原生 Qwen3

单 die eager 推理或多 die 独立副本。 [Qwen3 启动与验证](ascend.md#rsinfer-原生-qwen3)

### RSInfer 原生 GLM-5.3-Flash

W8A8 原生 worker、域内 TP/EP、MTP、ACL Graph 与跨重启缓存。 [GLM 启动与恢复](ascend.md#rsinfer-原生-glm-53-flash)

### vLLM Ascend GLM-5.3-Flash 对照

通过 `glm53` 子命令运行 vLLM Ascend，复用参考 hooks 或外部 MemCache 池。 [vLLM 对照部署](ascend.md#vllm-ascend-glm-53-flash-对照)

## Qwen3.8 与 DS1017

- `models/qwen38.py` 支持 `Qwen3_5ForConditionalGeneration` 和 `Qwen3_5ForCausalLM`。模型单元向 worker 提供 `model@1`。 [Qwen3.8 配方](models/qwen38.py)
- 4×SM100 上的 DS1017 用 `deploy/serve_ds1017.sh` 对齐当前 Python 的 CUDA 工具链。将 TileLang、DeepGEMM 等算子库安装到实际 worker 的 Python 环境。后端、分页和 YoCo 的选项见模型配方。 [SM100 启动与热更](../doc/quickstart_zh.md) · [DS1017 配方](models/ds1017.py)

## 命令

下表中的子命令接在 `python deploy/rsinfer.py` 后。从 `--run` 选择目标实例，默认目录为 `run/rsinfer`。

| 子命令 | 用途 |
| --- | --- |
| `serve <模型目录>` | 按模型生成拓扑并启动前端、调度和 worker。`--dry-run` 写出拓扑与配置并打印命令，`--detach` 在引擎就绪后返回。 |
| `status --json` | 查看健康、已提交 identity、进程状态及每个 worker 的 `local_pid`。 |
| `chat "…"` / `complete "…"` | 向运行目录记录的 HTTP 地址发送流式请求。 |
| `control check` / `control apply` | 检查源码与清单，预演或提交变更。`control logs` 查事务，`control introspect` 查各进程的当前状态。 [热更与恢复](../.claude/skills/rsinfer/SKILL.md#在跑着的引擎上热改) |
| `stop` | 向目标运行目录的进程发送终止信号并等待退出，默认超时 60 秒。 |
| `sleep` / `wake` | 暂时释放设备内存，再从当前 checkpoint 恢复服务。 |
| `checkpoint --config <目录> --out <目录>` | 用支持生成权重的模型配方创建随机 checkpoint，例如 CPU 调试使用的 Qwen3.8。 |
| `cache-save --out <清单路径>` | 将原生 GLM 可复用的前缀保存到 MemCache 并导出重启清单。 [保存与恢复](ascend.md#原生-memcache-跨重启) |

每个原生引擎使用独立的运行目录。`engine.json` 记录服务和控制面地址，`control/` 保存已提交的事务与快照信息，`host-a/snapshots/.run/*/log` 保存各进程日志，后台启动时还有 `engine.log`。启动时写入的 `config.json` 可用于检查初始配置，热更后的配置请按已提交的 identity 核对。

### 睡和醒（2026-11-11）

`sleep --wait-empty 120` 可等待在途请求跑完。睡眠期间暂停接收推理请求，worker 释放权重并缩小状态池。`wake` 从 `weights.path` 重载权重，按当前配方重建状态与图。恢复后重放同一请求。 [状态与权重变更](../.claude/skills/rsinfer/SKILL.md#状态权重与并行变更)

## 几个参数

| 参数 | 核对要点 |
| --- | --- |
| `--source`、`--set KEY=VALUE` | 源码目录须包含 `engine/`。`--set` 的值优先按 JSON 解析。 |
| `--weights`、`--tokenizer`、`--layers`、`--dummy` | 可指定权重和 tokenizer 目录，或按配方截层。Qwen3 缺少权重时须显式选 `--dummy` 才用随机权重，原生 GLM 需要 W8A8 checkpoint。 |
| `--npu` / `--npus`、`--gpu` / `--gpus` | 使用目标设备的物理编号。调度域与模型分片由配方决定。`--ranks` 可用于单设备或 CPU 上的多 rank 调试。 |
| `--host`、`--port`、`--control-host` | 前两个指定 HTTP 监听地址和端口，最后一项指定控制面和节点代理的地址，默认本机回环。 |
| `--mem-fraction`、`--max-context`、`--max-rows`、`--token-budget` | 分别控制设备内存份额、单请求总 token 上限、并发请求行数和每步 prefill token 预算。检查实际状态页与槽容量。 |
| `--overlap`、`--disable-radix-cache`、`--speculative-algorithm` | 调度重叠、前缀复用和投机解码会影响测量口径。按模型支持范围选择，记录实际生效的配置。 |
| `--timeout` | 原生启动等待和前端响应超时默认 3600 秒。读权重的变更还受 `change.execute_timeout_s` 约束。 |

## 跑过的（2026-11-09，当时的命令是 `deploy/qwen38.py serve --model <目录>`）

以下是 Qwen3.8 在 CPU、RTX 4090 D 和 SM100 上的早期验证记录。命令拼写与默认参数以本文前面的 CLI 和各配方为准。

### CUDA Graph 参数（2026-11-10）

默认 `--cuda-graph off`：worker 在 `torch_eager` 平台上，全部步都是 eager。`auto` 和 `required` 让 worker 用 `cuda_graph` 平台，给 decode 步录图；prefill 和 mixed 步仍然是 eager。

```sh
$A serve /path/to/model --cuda-graph auto
$A serve /path/to/model \
  --cuda-graph required --max-rows 16 --graph-buckets 1 2 4 8 16
```

- `auto`：能录就录；录不了（某个 decode kernel 不可捕获，或者图留下的显存超过 `--graph-memory-bytes`）就走 eager，原因在 worker 日志 `core.plan: {...}` 的 `cuda_graph.reason` 里。`required`：这些情况下启动失败，运行时没有图能装下的 decode 步也被拒绝。CPU 上 `auto` 是 eager，`required` 启动失败。
- `--graph-buckets`：每张图的行数，默认是不超过 `min(--max-rows, 8)` 的 2 的幂；行数更多的步走 eager。`required` 要求最大的桶等于 `--max-rows`，行数多于 8 时要显式列出。
  - 2026-11-11 起，回放的步用 paged attention 时（bf16 加 FlashInfer）默认值不同：SGLang 的档位 1、2、4、8、12，然后每 8 行一档，到 `min(--max-rows, 64)`，并包含这个上限本身。所以 `--max-rows` 不超过 64 时 `required` 不用再显式列桶；超过 64 时更多行的步走 eager，`required` 仍要显式列出包含 `--max-rows` 的桶。上面那条只适用于回放的步还在用固定容量 attention 的情况（fp32、KV 和模型 dtype 不同、没有 FlashInfer）。
- `--graph-memory-bytes`：全部图可以留下的设备显存，默认 1 GiB，来自 `--mem-fraction` 留给激活的那部分；状态池的大小与不开 graph 时相同。实际留下多少在日志的 `cuda_graph.kept_bytes`。
- 回放的 decode 步用的是固定容量的 attention（`kernels.torch_graph`），代价随 `--max-context` 增长而不是随序列的实际长度；4090 D 上 4 层模型，8 行在 4096 时比 eager 快 1.1–1.2 倍，在 16384 时比 eager 慢。eager 的 decode 步仍然用参照实现。数字见 [cuda-graph-review-fixes.md](../doc/evaluations/cuda-graph-review-fixes.md)，设计见 [cuda-graph.md](../doc/specs/cuda-graph.md)。完整 27B 上改过以后还没有重新验收。
- 2026-11-11：bf16 模型在装了 `gpu` extra 的机器上，attention 和 GDN 默认用真 kernel（FlashInfer、Triton），eager 和回放的步是同一套，回放的代价不再随 `--max-context` 增长；上一条只适用于 fp32 或没有 FlashInfer 的情况。FlashInfer 第一次 plan 要编译，`PATH` 上要有 `ninja` 和 CUDA 编译器：用 `uv run` 起，或者把环境的 `bin` 加进 `PATH`。见 [real-kernels-4090d.md](../doc/evaluations/real-kernels-4090d.md)。
- 2026-11-11：有 CUDA 设备时 worker 还会加载 `kernels.cuda`，它在加载时编译一个 C++/CUDA kernel（KV 写入）。一台机器上第一次启动因此多等约一到两分钟（4090 D 约 55 秒，SM100 / CUDA 13 约 123 秒），之后从 `~/.cache/torch_extensions` 加载；和 FlashInfer 一样要求 `PATH` 上有 `ninja` 和 CUDA 编译器。改了它的 C++ 源码以后直接 `control apply` 就行：编译在变更的准备阶段另起进程做，引擎照常服务（2026-11-11 起）。
- 改 `plan.*` 的配置或 kernel 单元的代码会在空引擎上重载 `core.plan` 并重录；范围以 `apply --dry-run` 为准。

### 原 eager 验证记录

- **CPU**（`tests/scenarios/test_m2a.py`，进 CI）：
  - 小随机模型，用 `checkpoint` 写成多模态布局；tokenizer 是 `tests/units/support/tokenizer.py` 生成的真 HF tokenizer。
  - `serve` 起整个引擎，completions、chat 和 `chat` 命令的流式输出，文本都和 HF 的贪心生成逐字相同，开、不开重叠都一样。
- **RTX 4090 D（24 GB），Qwen3.8-27B 的真实结构**，都是手动跑的 `tests/gpu/`：
  - **4 层，worker 栈对照 HF**（`test_qwen38_partial.py`）：用 `checkpoint` 生成的随机权重，7.6 GiB。
    - fp32：3 个 prompt 各贪心 12 个 token，和 HF 全部相同，prefill logits 误差在 1e-3 以内。
    - bf16：12 个 token 也全部相同，prefill logits 的余弦相似度 0.9996。
    - 两种精度下池子大小都由 holder 定。
  - **4 层，整个引擎对照 HF**（`test_qwen38_serve.py`）：
    - 用 Qwen3.6-27B 的真 tokenizer，`serve` 起引擎，经 HTTP 拿到的文本在 fp32 下和 HF 逐字相同，开、不开重叠都一样。
    - bf16 下有一个 prompt 第一个 token 就不同。fp32 下 HF 和我们给出同一个 token，所以这是两个候选 token 太接近、bf16 舍入不同造成的，不是 bug。
  - **8 层，dummy 权重**：用 `serve` 加 `chat`，chat 和 completions 都能跑通。
    - 1001 个 token 的 prefill 用 0.14 秒；
    - 单个请求 decode 每秒约 55 个 token（参照 kernel）；
    - 8 个请求并发（prompt 长度不同，贪心和采样混着）一起完成，共 2.2 秒。
  - **不重启，热加载改动**（RSI）：在跑着的引擎上改工作区，再用 `python -m core.control.client apply --source <工作区>`。
    - 2026-11-09 起，apply 时引擎必须是空的（[empty-apply.md](../doc/specs/empty-apply.md)）。默认有请求时被拒，提示里写着两条命令：`--wait-empty [秒]` 关门、等里面的请求跑完再 apply；`--force` 立刻结束它们再 apply。只改 live 键（如 `sched.overlap`）不受影响。apply 默认只打印摘要，`--json` 打印完整结果。下面"改动期间在跑的请求"的几条是这之前的记录。
    - 重载 `core.sched`、`layers.sampler`，以及改 `sched.overlap` 这个 live 配置：都在 2 秒内提交。改动期间在跑的长请求，输出和不改时逐字相同。
    - 重载 `frontend.detok`：提交成功，但正在进行的 HTTP 请求被断开，回了 500。原因是单元运行时会连同使用者（`frontend.http`）一起重载，HTTP 服务就跟着重启了。这是 `engine/frontend/README.md` 里"还没做"的那一条。
- **注意**：
  - 停引擎（Ctrl-C 或 SIGTERM）在 GPU 上要 60～90 秒，原因还没查。各进程是并行停的（`Supervisor.stop_all`），哪一段最费时，记录在 PR #32 的评论里。
  - 命令行退出时会带走引擎：起子进程时设了 `PR_SET_PDEATHSIG`，就算命令行被 SIGKILL，引擎也会收到 SIGTERM，不会留下占着显存的进程。
  - `checkpoint` 生成 4 层的随机权重要 7.6 GiB 磁盘，跑完记得删掉。
- **完整的 27B 模型**（真权重，SM100，TP1，参照 kernel，eager；2026-11-09，#37）：
  - `tests/gpu/test_qwen38_serve.py`：bf16 和 f32 各 2 passed（开、不开重叠）；f32 下经 HTTP 的文本和 HF 逐字相同。`test_qwen38_partial.py` 2 passed。
  - 常驻 `serve`（bf16，默认参数）：
    - 约 37 秒到 "engine up"（权重已在页缓存里；冷读 `/shared-fs` 会更慢）；
    - holder：`101.36 GiB for states: gdn.slots 334 (47.89 GiB), kv.paged 54753 (53.47 GiB)`，即约 87.6 万个 token 的 KV；GPU 上共用 155992 MiB；
    - `core.plan` 的记录：attention 和 GDN 都是 `reference`（fallback），KV bf16、页 16，conv bf16，temporal f32；
    - prefill 1081 个 token 0.564 秒（约 1900 tok/s）；单个请求 decode 11.9 tok/s（约 84 ms 一个 token）；8 个请求并发、各 128 个 token，一起在 18.0 秒完成，合计 56.9 tok/s。
  - 不停引擎热改（请求生成到一半时 `apply`）：改 `policies.fcfs` 的代码、改 `kernels.reference` 的 `ops.py`、改 `layers.attn` 的代码（连带重载模型、`core.plan`、executor、`core.worker`）、改 `attn.decode_backend`、改 live 键 `sched.overlap`，都在 1.2–1.8 秒内提交。暂停（pause）约 0.09 秒，请求看到的最大 token 间隔 0.21–0.23 秒（平时 0.09 秒），输出和不改时逐字相同。改 `kernels.reference` 以后，backend 在下一步重新解析了 kernel（worker 日志里的 `layers: kernels resolved again`）。改 `state.kv.dtype` 被拒，要求重启引擎。
  - 这次修掉的问题：
    - `--gpu` 不给时，worker 总在 GPU 0 上：节点代理只按拓扑给引擎进程设 `CUDA_VISIBLE_DEVICES`，调用方自己设的被丢掉，同一台机器起两个引擎就挤在同一张卡上。现在 `--gpu` 默认取调用方 `CUDA_VISIBLE_DEVICES` 的第一张卡（`worker_gpu`），没设才是 0。
    - 要求重启引擎时，诊断一律说 "the change touches the framework core"，连 `state.kv.dtype` 这种"会丢在途状态"的情况也这么说（真正的原因在同一个结果的 `plan.notes` 里）。改法要动框架本体（`core.change` 的协调者），另开 PR 到 master（#39），不在这里。
    - 调度宿主在没有请求时重启，引擎就一直不接活（Codex，#37）：worker 只在结果连接换了时补发 `Capacity`，而连接要在写消息时才发现换了，空闲的 worker 不写消息；新调度器等容量，worker 等计划。现在空闲的 worker 每 `worker.announce_every_s`（1 秒）补发一次 `Capacity`。
    - `--model A --tokenizer B` 时丢了模型的结束 token（Codex，#37）：结束 token 只从 tokenizer 目录 B 的配置里读，A 声明的（比如 `[0, 5]`）丢了，模型输出结束符以后还接着生成。现在照 SGLang 从模型目录读（`frontend.model_path`），再并上 tokenizer 自己的。
    - f32 和 HF 对比时，cuDNN 的卷积默认走 TF32，部分模型的首步 logits 差到 8e-3：GPU 测试现在关掉 TF32（`tests/gpu/support.py`）。
    - conv state 默认 bf16（照 SGLang，#35），f32 对比要加 `--conv-dtype f32`；`serve` 新增 `--kv-dtype`、`--ssm-dtype`、`--conv-dtype`。
