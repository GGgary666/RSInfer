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
- `--device cpu`：整个引擎跑在 CPU 上（`tests/scenarios/test_m2a.py` 就是这样跑的）。

## 跑过的（2026-11-09）

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
    - 重载 `core.sched`、`layers.sampler`，以及改 `sched.overlap` 这个 live 配置：都在 2 秒内提交。改动期间在跑的长请求，输出和不改时逐字相同。
    - 重载 `frontend.detok`：提交成功，但正在进行的 HTTP 请求被断开，回了 500。原因是单元运行时会连同使用者（`frontend.http`）一起重载，HTTP 服务就跟着重启了。这是 `engine/frontend/README.md` 里"还没做"的那一条。
- **注意**：
  - 停引擎（Ctrl-C 或 SIGTERM）在 GPU 上要 60～90 秒，原因还没查。各进程是并行停的（`Supervisor.stop_all`），哪一段最费时，记录在 PR #32 的评论里。
  - 命令行退出时会带走引擎：起子进程时设了 `PR_SET_PDEATHSIG`，就算命令行被 SIGKILL，引擎也会收到 SIGTERM，不会留下占着显存的进程。
  - `checkpoint` 生成 4 层的随机权重要 7.6 GiB 磁盘，跑完记得删掉。
- **完整的 27B 模型**（真权重，SM100）：交给 Devin 跑，交接说明在这个 PR 的评论里。
