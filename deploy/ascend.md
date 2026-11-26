# Ascend 910C 单机部署

从仓库根目录执行。先用 `npu-smi info -m` 确认物理 die，用 `npu-smi info` 确认占用。下列模型目录、die 和端口是示例值，请换成本机资源。为每个服务使用独立运行目录。同一组 die 上的原生与对照服务顺序运行。 [设备预检](../doc/2026-11-26-ascend-npu-workflow.md#1-核对设备和运行环境) · [文档导航](../doc/README.md)

## 安装依赖

910C 实测环境为 Linux aarch64、Python 3.12、CANN 9.2.0、PyTorch 2.10.0 和 TorchNPU 2.10.0.post7。准备 `uv`、主机驱动和固件，以及与设备匹配的 CANN Toolkit 和 OPP。在启动 worker 的 shell 加载所装版本的环境脚本。 [TorchNPU 配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md) · [910C 实测环境](../doc/rsi/2026-11-26-ascend-a3.md)

```sh
. "${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/ascend-toolkit}/set_env.sh"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

### Qwen3 依赖

`ascend` extra 安装 RSInfer 的 HTTP 前端、权重读取、TorchNPU 和 TBE 依赖。它从 PyTorch CPU 索引取得 2.10.0，由 TorchNPU 注册 NPU 后端。`ascend`、`cpu` 和 `gpu` 每个环境选一个。把示例中的临时环境路径换成部署使用的持久路径，后续启动和热更继续用这个 Python。

```sh
RSI_ASCEND_VENV="$(mktemp -d)/venv"
UV_PROJECT_ENVIRONMENT="$RSI_ASCEND_VENV" uv sync --locked --python 3.12 --no-dev --extra ascend
. "$RSI_ASCEND_VENV/bin/activate"
python -c 'import torch, torch_npu, tbe; print(torch.__version__, torch_npu.__version__, tbe.__file__)'
```

### GLM 与 MemCache 依赖

原生 GLM 和 vLLM 对照都使用带编译产物的 A3 环境。已有参考环境可以直接复用。新部署可从 `quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` 开始。实测使用 vLLM 0.30.0、vLLM Ascend 0.30.0rc2、A3 FLA-NPU `26.9.1+deva4a7958` 和 Triton Ascend `3.6.0+dev20260923230212`。镜像提供 GLM 的 CANN 自定义算子和 OPP。 [A3 镜像构建配方](https://github.com/vllm-project/vllm-ascend/blob/v0.30.0rc2/Dockerfile.a3) · [A3 设备挂载方式](https://docs.vllm.ai/projects/ascend/en/v0.30.0/getting_started/installation.html)

在这个环境里用一个继承镜像 Python 包的虚拟环境安装 RSInfer 基础依赖。下面的 `uv sync` 仅选择项目基础依赖，沿用镜像提供的 PyTorch、前端与算子。

```sh
RSI_GLM_VENV="$(mktemp -d)/venv"
uv venv --system-site-packages --python "$(command -v python3)" "$RSI_GLM_VENV"
. "$RSI_GLM_VENV/bin/activate"
uv sync --locked --active --inexact --no-dev
python -c 'import torch_npu, fastapi, jinja2, safetensors, transformers, uvicorn; import vllm_ascend.vllm_ascend_C, fla_npu.ops.ascendc'
```

启用外部池或使用 `--memcache-client-config` 时，worker 和池都要能加载 MemCache Hybrid 与 MemFabric Hybrid。参考镜像自带 1.3.0 配对。需要补装 SDK 时，在上面的 GLM 虚拟环境执行 `uv sync --locked --active --inexact --no-dev --extra ascend-memcache`。在各自的启动环境检查 `python -c 'from memcache_hybrid import DistributedObjectStore, MetaService'`。池地址与配置仍按下面的启动流程准备。 [MemCache 软件与配置](https://docs.vllm.ai/projects/ascend/en/v0.30.0/user_guide/feature_guide/kv_cache_pool_ascend_store.html)

## RSInfer 原生 Qwen3

单 die 可以完成第一条推理请求。使用含 `config.json`、tokenizer 和 safetensors 权重的 Qwen3 目录。

```sh
QWEN_MODEL=/path/to/Qwen3-0.6B
QWEN_RUN=/tmp/rsinfer-qwen3
QWEN_DIE=0
QWEN_PORT=18650
export OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0

python deploy/rsinfer.py serve "$QWEN_MODEL" --device npu --npu "$QWEN_DIE" \
  --port "$QWEN_PORT" --max-context 4096 --max-rows 16 --token-budget 512 \
  --mem-fraction 0.08 --run "$QWEN_RUN" --detach
python deploy/rsinfer.py status --run "$QWEN_RUN" --json
python deploy/rsinfer.py complete 'The capital of France is' --run "$QWEN_RUN" --max-tokens 16
```

当前 NPU 配方使用 eager 和默认 128-token KV 页。换成 `--npus 0,1` 可在两个 die 各放一份完整模型，独立接请求。确认输出后可运行流式基准，结束时用 `python deploy/rsinfer.py stop --run "$QWEN_RUN"` 停服。 [Qwen3 基准与输出核对](../doc/2026-11-26-ascend-npu-workflow.md#qwen3-流式基准)

## RSInfer 原生 GLM-5.3-Flash

准备完整 W8A8 checkpoint，激活上面的 GLM 环境。下面的命令使用一个 TP8/EP8 调度域，每个 die 一个 worker。函数保留启动参数，供预览、启动及缓存恢复复用。

```sh
GLM_MODEL=/path/to/GLM-5.3-Flash-w8a8
GLM_DIES=0,1,2,3,4,5,6,7
GLM_RUN=/tmp/rsinfer-glm53-native
GLM_PLAN=$(mktemp -d)
GLM_PORT=18651
export OMP_NUM_THREADS=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0

glm_native() {
    python deploy/rsinfer.py serve "$GLM_MODEL" --device npu --npus "$GLM_DIES" \
      --port "$GLM_PORT" --max-context 65536 --max-rows 32 --max-slots 160 \
      --token-budget 8192 --set state.kv.pages=3072 --mem-fraction 0.89 \
      --speculative-algorithm EAGLE --speculative-num-steps 4 \
      --speculative-num-draft-tokens 5 --acl-graph required "$@"
}
glm_native --run "$GLM_PLAN" --dry-run
glm_native --run "$GLM_RUN" --detach
python deploy/rsinfer.py status --run "$GLM_RUN" --json
python deploy/rsinfer.py complete '1+1=' --run "$GLM_RUN" --max-tokens 16
```

`$GLM_PLAN/topology.json` 给出 rank 放置和 mesh，`config.json` 给出 `sched.domains` 和 `plan.cuda_graph`。在实际 worker 中，prefill 走 eager，decode 默认要求 ACL Graph。启用示例中的 MTP 后，每轮最多提议 4 个 token，验证窗口为 5。用目标请求的 trace 检查 draft 和 verify 的实际回放。代码热更、排空、回退及实例 PID 核对见 [原生热更流程](../.claude/skills/rsinfer/SKILL.md#在跑着的引擎上热改)。

要部署两个 TP8/EP8 域，将 `GLM_DIES` 设为 16 个物理 die，并在预览和启动命令中都加 `--glm-dp-size 2`。启用 DFC 需要每域至少两个 rank。跨域前缀复制还需要多个域和 hybrid radix 缓存。复现某轮结果时，沿用该轮的 MTP、DFC、调度和模型配置。 [整机复现参数](../doc/rsi/2026-11-26-glm53-native-rsi-iter.md#热更和复现参数) · [跨域前缀复制](../doc/rsi/2026-11-26-glm53-native.md#跨域前缀复制)

### 原生 MemCache 跨重启

先准备运行中的外部池、客户端 DRAM 为 0 的配置 `MMC_CLIENT` 和固定的请求文件 `CACHE_REQUEST`。请求使用 `GLM-5.3-Flash`、足以覆盖完整 KV 页的文本前缀，以及固定的 `cache_salt`。在上例首次启动时改用下面带 `--memcache-client-config` 的启动命令。若复现多域或其他已调优配置，保存与恢复两次启动都带上相同的附加参数。

```sh
PREFIX_MANIFEST="$GLM_RUN/prefix-manifest.json"
replay_prefix() {
    curl -fsS -H 'Content-Type: application/json' --data-binary "@$CACHE_REQUEST" \
      "http://127.0.0.1:$GLM_PORT/v1/completions"
}
glm_native --run "$GLM_RUN" --memcache-client-config "$MMC_CLIENT" --detach
replay_prefix > "$GLM_RUN/cache-cold.json"
python deploy/rsinfer.py cache-save --run "$GLM_RUN" --out "$PREFIX_MANIFEST"
python deploy/rsinfer.py stop --run "$GLM_RUN"
glm_native --run "$GLM_RUN" --memcache-client-config "$MMC_CLIENT" \
  --prefix-manifest "$PREFIX_MANIFEST" --detach
replay_prefix > "$GLM_RUN/cache-restored.json"
```

保留原池，保持模型、状态格式与 rank 划分兼容。重启后核对新 worker PID，重放同一请求并验证缓存读取与输出。测试结束可用同一 `stop --run` 命令关闭原生引擎。 [恢复验收](../doc/2026-11-26-ascend-npu-workflow.md#memcache-跨重启验收)

## vLLM Ascend GLM-5.3-Flash 对照

`glm53 serve` 使用 vLLM Ascend 运行同一 W8A8 模型。下例用 16 个 die、DP2 × TP8、EP、MTP5 和 ACL decode graph。设置 `GLM_MODEL` 为权重目录，`REF_DIES` 为与 DP × TP 数量一致的物理 die 列表。

```sh
GLM_MODEL=/path/to/GLM-5.3-Flash-w8a8
GLM_REF_RUN=/tmp/rsinfer-glm53-vllm
REF_DIES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
python deploy/rsinfer.py glm53 serve "$GLM_MODEL" --npus "$REF_DIES" \
  --dp 2 --tp 8 --max-batched-tokens 8192 --max-seqs 32 --port 18652 --run "$GLM_REF_RUN"
python deploy/rsinfer.py glm53 status --run "$GLM_REF_RUN"
```

需要复现带 hooks 和缓存的对照时，在启动命令上加 `--hooks-dir "$REF_HOOKS"`（含 `sitecustomize.py`）。用 `--memcache-meta-config "$MMC_META" --memcache-pool-config "$MMC_POOL"` 创建由该运行目录管理的池，或用 `--memcache-client-config "$MMC_CLIENT" --memcache-metrics-url "$MMC_METRICS_URL"` 加入外部池。配置文件和可达的服务地址由运行环境提供。 [参考部署与缓存实测](../doc/rsi/2026-11-26-ascend-a3.md#glm-53-flash-与-memcache)

检查 `launch.json` 的拓扑和参数，`ready.json` 的服务就绪、首请求及池状态。用 `python deploy/rsinfer.py glm53 restart --run "$GLM_REF_RUN" --max-batched-tokens 8192` 重启推理进程。托管池的复用上限由 `--memcache-max-starts` 控制，默认 12 次启动，达到上限后下次启动会重建。普通 `glm53 stop --run "$GLM_REF_RUN"` 保留池，回收该实例管理的池与 MetaService 时加 `--with-memcache`。 [原生与 vLLM 的 AISBench 对照](../doc/2026-11-26-ascend-npu-workflow.md#glm-aisbench-对照)
