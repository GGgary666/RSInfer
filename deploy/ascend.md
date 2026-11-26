# Ascend 910C 单机部署

在仓库根目录使用已加载 CANN 的 Python。先用 `npu-smi info -m` 确认物理 die，用 `npu-smi info` 确认占用。下列模型目录、die 和端口是示例值，请换成本机资源。为每个服务使用独立运行目录。同一组 die 上的原生与对照服务顺序运行。 [环境预检](../doc/2026-11-26-ascend-npu-workflow.md#1-核对设备和运行环境) · [文档导航](../doc/README.md)

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

准备完整 W8A8 checkpoint，并在 worker 环境安装带自定义算子的 `vllm_ascend` 和 `fla_npu`。下面的命令使用一个 TP8/EP8 调度域，每个 die 一个 worker。函数保留启动参数，供预览、启动及缓存恢复复用。

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
