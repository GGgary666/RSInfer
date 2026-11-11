# 部署：在一台机器上跑引擎（2026-11-11 起是 `agxinfer.py`）

`deploy/` 放的是不属于 pytest 的部署工具。用户同意为它新开这个顶层目录。

- `agxinfer.py`：命令行。2026-11-11 取代了 `qwen38.py`：命令按"引擎的生命周期"和"改跑着的引擎"分开，不再绑死一个模型。
- `models/`：每个模型家族一个 recipe，按模型目录的 `config.json` 选。现在只有 `qwen38.py`（`Qwen3_5ForConditionalGeneration` / `Qwen3_5ForCausalLM`）。`hf.py` 是几个 recipe 共用的读配置、截层。

## 命令

`A="uv run python deploy/agxinfer.py"`（GPU 环境不是 uv 管的时候，用那个环境的 `python`）。

| 命令 | 做什么 |
|---|---|
| `$A serve <模型目录>` | 起一个引擎：前端、调度宿主、一个 worker（TP1），经过 `python -m core.control serve --spawn-agents`。默认在前台跑到 Ctrl-C；`--detach` 在后台跑，引擎起来就返回；`--dry-run` 只写拓扑和配置、打印命令 |
| `$A status` | 引擎在不在跑、URL、控制面地址、身份、各进程状态、`/health` |
| `$A stop` | 停引擎：给控制面发 SIGTERM，等这个引擎的进程全部退出；到 `--timeout`（默认 60 秒）还没退的 kill 掉 |
| `$A sleep` / `$A wake` | 让出显存 / 拿回来（`holder.sleep`，见下面「睡和醒」） |
| `$A control <命令> ...` | 控制面客户端（`python -m core.control.client`）：`check`、`apply`、`status`、`introspect`、`logs`，参数照旧；控制面地址从运行目录里找 |
| `$A chat "..."` / `$A complete "..."` | 发一个请求（chat 或 completions），流式打到标准输出 |
| `$A checkpoint --config <目录> --out <目录>` | 按一份配置生成随机权重，布局和这个模型家族的真 checkpoint 一样（Qwen3.8：多模态命名 `model.language_model.*`，外加加载器会跳过的 `visual`、`mtp` 张量）；可以只保留前 N 层 |

**运行目录**（`--run`，默认 `run/agxinfer`）就是一个引擎：拓扑和配置（`topology.json`、`config.json`）、控制面的状态（`control/`）、快照和各进程的日志（`host-a/snapshots/`）、`engine.json`（控制面进程、URL、控制面地址），`--detach` 时还有 `engine.log`。每个命令都靠 `--run` 找到它的引擎；一个运行目录里同时只能有一个引擎。

```sh
uv sync --python 3.12 --extra gpu --extra frontend      # 没有 GPU 的机器用 --extra cpu

$A serve /path/to/Qwen3.8-27B --detach                  # 真权重（SM100，完整模型）
$A serve /path/to/Qwen3.8-27B --layers 8 --dummy        # 只要前 8 层，dummy 权重
$A status
$A chat "你好"
$A complete "The capital of France is"
$A control apply --source . --dry-run                   # 改了工作区：看计划
$A control apply --source . --intent "为什么改"
$A stop
```

和 vLLM、SGLang 的对应：`serve` 的模型目录是位置参数（`vllm serve <model>`），`--served-model-name`、`--host`、`--port` 同名；`--mem-fraction` 是 SGLang 的 `--mem-fraction-static`，`--kv-dtype` 是 `--kv-cache-dtype`。改跑着的引擎是我们自己的事，放在 `control` 下面。

和 `qwen38.py` 比改了的：

- 模型目录从 `--model` 改成位置参数（`--model` 还认）；
- `--bind` 改叫 `--host`，`--name` 改叫 `--served-model-name`（旧名字还认）；
- `plan` 并进了 `serve --dry-run`，`chat --completion` 改成 `complete`；
- 新增 `--detach`、`--set 键=值`（任何配置键，值按 JSON 解析，解析不了当字符串）、`status`、`stop`、`sleep`、`wake`、`control`。

### 睡和醒（2026-11-11）

`sleep` 是一次变更，把 `holder.sleep` 设成 true：

- worker 把权重换成不占内存的 meta 张量，KV 和 GDN 状态池缩到每种一项，不录 graph；
- 调度器拒绝所有请求，回 400，说引擎睡着了。

`wake` 把它改回 false：从 checkpoint 重新读权重，重建状态池，重录 graph（和 vLLM 第 2 级的 sleep 一样不留权重的副本，`refit` 换过的权重回到 checkpoint）。

4090 D 上 4 层的引擎：21,482 MiB 降到 962 MiB，剩下的是 CUDA 上下文和 kernel 库的工作区。睡 2 秒，醒 3.6 秒，醒来输出和睡前逐字相同。睡着的时候 `tests/gpu/` 的八个文件都能在同一张卡上跑。记录见 [recovery-sleep-cli.md](../doc/evaluations/recovery-sleep-cli.md)。

- 和别的变更一样要求引擎是空的，`sleep` 收 `--wait-empty`、`--force`。
- 醒的时候读权重在暂停期里，受执行时限（30 秒）约束；完整 64 层模型上还没试过。

## 几个参数

`serve` 的参数分组见 `$A serve --help`。


- `--layers N`：只保留配置里的前 N 层。和完整模型一样，每 4 层里有 1 层全注意力。用于完整模型放不下的时候。
- 权重：
  - 默认用模型目录里的 `*.safetensors`，也可以用 `--weights` 指定别的目录；
  - `--dummy`，或者目录里没有权重文件时，用 dummy 权重（`initialize_dummy_weights`）。
- `--tokenizer`：tokenizer 不在模型目录里时用。
- 显存：
  - 状态池由 `core.state.holder` 分：`--mem-fraction`（默认 0.85，对应 SGLang 的 `mem_fraction_static`）；`--max-slots`（对应 `max_mamba_cache_size`）可以直接指定 GDN 槽数。
  - `--max-rows`：最多同时跑多少个请求（`batch.max_rows`，也是 `sched.max_batch`）。
  - `--max-context`：一个请求最长多少 token，prompt 加输出（`batch.max_context_len`）。
- `--timeout`（默认 3600 秒）：
  - 启动时等进程签到、等启动变更准备，各用这个时长。单元会在签到之前加载权重，大模型要几分钟。
  - 前端两次响应之间最长的沉默，也是这个时长。
- `--overlap`：一步重叠（`sched.overlap`）。
- `--gpu`：worker 用哪张卡（物理编号）。不给时取 `CUDA_VISIBLE_DEVICES` 的第一张，没设就是 0。节点代理只按拓扑设卡，所以靠 `CUDA_VISIBLE_DEVICES` 选卡时也是经这里生效。
- `--device cpu`：整个引擎跑在 CPU 上（`tests/scenarios/test_m2a.py` 就是这样跑的）。
- `--set 键=值`：没有专门参数的配置键，可以重复。

## 跑过的（2026-11-09，当时的命令是 `deploy/qwen38.py serve --model <目录>`）

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
- 2026-11-11：有 CUDA 设备时 worker 还会加载 `kernels.cuda`，它在加载时编译一个 C++/CUDA kernel（KV 写入）。一台机器上第一次启动因此多等约一到两分钟（4090 D 约 55 秒，SM100 / CUDA 13 约 123 秒），之后从 `~/.cache/torch_extensions` 加载；和 FlashInfer 一样要求 `PATH` 上有 `ninja` 和 CUDA 编译器。
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
