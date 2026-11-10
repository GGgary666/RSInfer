# 部署：在一台机器上跑 Qwen3.8（2026-11-09）

`deploy/` 放的是不属于 pytest 的部署工具。用户同意为它新开这个顶层目录。

`qwen38.py` 是一个命令行工具，在一台机器上起整个引擎：前端、调度宿主、一个 worker（TP1），中间经过 `python -m core.control serve --spawn-agents`。它有四个子命令：

| 子命令 | 做什么 |
|---|---|
| `plan` | 按模型目录生成拓扑和配置（`topology.json`、`config.json`，写进 `--run` 目录），并打印对应的 `serve` 命令 |
| `serve` | 先做 `plan`，再起引擎，一直跑到 Ctrl-C。引擎起来后打印前端地址 |
| `chat` | 发一个请求（chat，或者加 `--completion` 发 completions），把流式结果打到标准输出 |
| `checkpoint` | 按一份配置生成随机权重（HF 的初始化，带 seed），布局和 Qwen3.8 的真 checkpoint 一样：多模态命名 `model.language_model.*`，外加加载器会跳过的 `visual`、`mtp` 张量。可以只保留前 N 层 |

## 用法

```sh
uv sync --python 3.12 --extra gpu --extra frontend      # 没有 GPU 的机器用 --extra cpu

# 真权重（SM100，完整模型）
uv run python deploy/qwen38.py serve --model /path/to/Qwen3.8-27B
# 只要前 8 层，用 dummy 权重（显存不够放完整模型时）
uv run python deploy/qwen38.py serve --model /path/to/Qwen3.8-27B --layers 8 --dummy
# 另开一个终端
uv run python deploy/qwen38.py chat "你好"
uv run python deploy/qwen38.py chat --completion "The capital of France is"
```

## 几个参数

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

## 跑过的（2026-11-09）

### CUDA Graph 参数（2026-11-10）

CUDA 部署默认 `--cuda-graph auto`，使用 `cuda_graph` 平台；`--cuda-graph off` 使用原 `torch_eager`。CPU 的 auto 保持 eager，CPU 上 required 启动失败。

```sh
uv run python deploy/qwen38.py serve --model /path/to/model \
  --cuda-graph required --max-rows 4 --graph-buckets 1 2 4 \
  --max-context 512 --graph-memory-bytes 1073741824
```

只录每行一个 token 的 decode，prefill/mixed 走 eager。`--graph-buckets` 默认是不超过 `--max-rows` 的 2 的幂；required 必须覆盖最大行数（非 2 的幂需显式列入）。`--graph-memory-bytes` 是 holder 预留和 graph 构建峰值上限，默认 1 GiB。构建、校验或预算失败会终止启动，不会悄悄回退。

首版用 PyTorch 固定容量 SDPA 和 GDN 参照实现；内存/计算随 bucket 与最大上下文增长，大模型应从小容量开始验证。日志 `core.plan` 的 `cuda_graph` 字段给出实际 buckets、回退原因、构建耗时和峰值。完整 27B/SM100 的性能尚未验收，详见 [实现方案](../doc/specs/cuda-graph.md)。启动配置通过空引擎 `apply --config` 重载读取者及其使用者，必要时重建状态池和 plan；修改状态配置不必然要求重启进程，已验证 GDN 槽容量可原地重载并重录。范围以 dry-run 为准。第一次引入平台代码需要重启引擎。

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
