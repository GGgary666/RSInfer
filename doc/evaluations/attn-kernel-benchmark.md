# FlashInfer 与现有 attention 的算子对比（2026-11-11）

> **读之前先看这三点**（2026-11-11 审查后补的）
>
> 1. **这是接入之前做的单层微基准，测的接法不是后来引擎里用的那一种。** 所以这些数字回答的是"把按容量算的 attention 换成 paged kernel 值不值"，不能当成引擎里那条路径的数字。
>
>    | | 这里测的 | 引擎里实际用的（#51） |
>    |---|---|---|
>    | FlashInfer 版本 | 0.6.14 | 0.7.2 |
>    | 页表 | 页大小 16，列表里是页号 | 页大小 1，列表里是 token 槽号 |
>    | workspace | 每个实例 128 MiB | 每个设备 384 MiB，共用 |
>
>    引擎实际的做法见 [attn-kernel.md](../specs/attn-kernel.md)，端到端的数字见 [serving-flashinfer-gdn-ablation.md](serving-flashinfer-gdn-ablation.md)。
> 2. **脚本 `tests/gpu/bench_attn.py` 已经不在仓库里。** 它在本 PR 分支历史的提交 `03579e3` 里，要复现先取出来：`git show 03579e3:tests/gpu/bench_attn.py > tests/gpu/bench_attn.py`。
> 3. 文中说的"设计"指当时的提案（提交 `502f5ce` 里的 `doc/specs/attn-kernel.md`），不是现在那份文档。

本次通过 **ossrun** 把 `dev/gw/attn-kernel` 独立 worktree 的源码快照发布到远程，在 NVIDIA SM90-80GB 上测量单层 full-attention decode。测量时引擎还没有接入 FlashInfer；这里没有完整模型、HTTP 吞吐或 SM100 的结果。

## 环境和可追溯性

| 项目 | 实测配置 |
|---|---|
| GPU | NVIDIA SM90-80GB，SM90，物理 GPU 1，约 80 GiB |
| Python / PyTorch | 3.12.7 / 2.11.0+cu130 |
| CUDA runtime / driver | 13.0 / 580.105.08 |
| FlashInfer | 0.6.14，显式 `backend="fa2"`、`use_tensor_cores=True` |
| 相关包 | apache-tvm-ffi 0.1.13、nvidia-cutlass-dsl 4.8.1、cuda-python 13.5.0、ninja 1.13.2 |
| 算子形状 | BF16；24 query heads / 4 KV heads；head dim 256；page size 16 |
| 工作区 | 每个 FlashInfer 实例 128 MiB |
| 源码 | 分支 `dev/gw/attn-kernel`；测量时基于 commit `502f5ce68214388ddcb54ca22950a341fa6892cc`，额外包含当时未提交的 benchmark 和 skill 目录地图更新 |
| ossrun release | `20261111T035845Z_502f5ce-dirty_5ab4bf00` |
| 源码包 SHA256 | `ee2fd4d021829aa9b3029621f7c6dfb350936bfe9c0eaa281db6b9eae6f643e5` |
| benchmark SHA256 | `716934d0349b9b6f67b9ffa698068e4dcb2ab6b2ace9e5e133862d1dad8f9358` |
| 测量 command ID | `cmd_20261111T035912Z_attn-bench-matrix_b9d092f1` |

FlashInfer 及补充依赖安装在独立 target 目录，未修改远程原有 Python 环境或本仓库依赖。这里固定了可运行组合，尚未完成支持版本范围的验收。

## 测量口径

四条路径使用相同随机种子 17、相同 Q/K/V 和随机打乱的物理页。每个模式和形状使用独立子进程，执行顺序打乱。每项预热 5 次，测 7 组，每组连续执行 20 次；表中单位为 **μs，取各组平均单次 wall time 的中位数**。

- `reference_eager`：直接调用仓库的 `kernels.reference.ops.attention_decode`，含逐请求循环、长度读取、KV gather 和 PyTorch SDPA。
- `torch_graph`：仓库的 `kernels.torch_graph.ops.attention_decode`，capture 后 replay；对 `max_context` 的整个宽度执行 gather 和带 mask 的计算。
- `flashinfer_eager`：paged decode wrapper 的 `run`。
- `flashinfer_graph`：相同 wrapper 的 graph 模式，capture `run` 后 replay。

分别测两个范围：

1. `prepared_run`：已经完成元数据准备后的 attention 执行。排除外部 prepare、KV store；reference 算子内部原有的 host 工作仍然包含在内，因此它不是纯 GPU kernel 时间。
2. `prepare_store_run`：每次更新元数据，写入当前 token 的 K/V，再执行一层 attention。FlashInfer 每次执行 `plan`；page indices 从 pinned CPU buffer 异步拷到固定 GPU staging，CPU indptr/last-page-length 供 host planning。reference/torch graph 更新 rows、lengths、写入位置。

graph 模式分别捕获计算和 KV store + 计算两张图，两图顺序运行并共享内存池。prepared 和含 prepare 的两组测量分开执行，二者之差不是独立测得的 planning 时间。CUDA event 的 `stream_elapsed` 包含 host 提交间隙，也不能当成纯 kernel profiler 时间。JSON 中的 p90 是 7 个组平均值的分位数，不是请求尾延迟。

这是固定长度、重复访问同一份 KV 的热缓存微基准。未计入 CPU 构造页表、QKV projection、RoPE、调度、采样、GDN、HTTP，也没有真实请求连续增长。引擎里每步只 plan 一次，各 attention 层共用，所以不能把本次 `prepare_store_run` 直接乘以层数预测模型性能。

## 结果

### 每步 prepare + KV store + 一层 attention

主矩阵 `max_context=4096`，下表包括每步 planning；加速比小于 1 表示变慢。

| batch | 实际长度 | reference eager | torch graph | FlashInfer eager | FlashInfer graph | FI graph 相对 torch graph |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 116.2 | 58.1 | 95.1 | 62.1 | 0.94× |
| 1 | 256 | 111.0 | 57.0 | 104.0 | 60.4 | 0.94× |
| 1 | 4096 | 120.1 | 63.6 | 102.2 | 61.1 | 1.04× |
| 8 | 32 | 681.5 | 487.7 | 109.2 | 75.0 | 6.50× |
| 8 | 256 | 713.6 | 490.1 | 116.5 | 79.0 | 6.20× |
| 8 | 4096 | 749.8 | 528.8 | 129.7 | 90.7 | 5.83× |
| 32 | 32 | 2717.1 | 1791.7 | 157.0 | 125.5 | 14.28× |
| 32 | 256 | 2733.6 | 1810.7 | 170.6 | 124.2 | 14.58× |
| 32 | 4096 | 2885.9 | 1966.8 | 231.8 | 228.6 | 8.60× |

batch 8/32 下，FlashInfer graph 相对现有 torch graph 为 **5.83–14.58×**，相对 reference eager 为 **8.27–22.01×**。batch 1 下为 **0.94–1.04×**：较短序列的计算收益被准备开销抵消，长度 4096 时只略快。FlashInfer eager 相对 reference eager 的收益也随 batch 明显变化，不能仅凭选用了 FlashInfer 就承诺固定倍数。

### 已完成准备后的 attention 执行

| batch | 实际长度 | reference eager | torch graph | FlashInfer eager | FlashInfer graph | FI graph 相对 torch graph |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 32 | 89.7 | 45.0 | 15.5 | 13.5 | 3.32× |
| 1 | 256 | 86.3 | 44.7 | 20.0 | 15.8 | 2.83× |
| 1 | 4096 | 92.9 | 48.5 | 19.1 | 18.5 | 2.63× |
| 8 | 32 | 659.6 | 472.2 | 15.3 | 13.8 | 34.15× |
| 8 | 256 | 685.5 | 475.4 | 18.0 | 16.5 | 28.86× |
| 8 | 4096 | 722.9 | 513.6 | 57.5 | 58.3 | 8.81× |
| 32 | 32 | 2688.6 | 1776.6 | 15.2 | 14.8 | 119.67× |
| 32 | 256 | 2714.3 | 1794.3 | 18.8 | 19.5 | 91.91× |
| 32 | 4096 | 2847.1 | 1951.5 | 185.8 | 185.5 | 10.52× |

这里的 FlashInfer 在 batch 1 也明显快于现有计算路径，说明小 batch 的集成重点是准备和提交开销。batch 32、长度 4096 时，FlashInfer eager/graph 的计算耗时接近（185.8/185.5 μs）；graph 不会在已有单次高效 kernel 上再自动产生大幅加速。

### 容量和变长

固定 batch 8，对比实际长度不变、`max_context` 从 4096 增至 16384。以下仍为含 prepare 和 KV store 的耗时：

| 实际长度 | max_context | reference eager | torch graph | FlashInfer eager | FlashInfer graph |
|---:|---:|---:|---:|---:|---:|
| 256 | 4096 | 713.6 | 490.1 | 116.5 | 79.0 |
| 256 | 16384 | 690.0 | 1792.7 | 117.8 | 75.7 |
| 4096 | 4096 | 749.8 | 528.8 | 129.7 | 90.7 |
| 4096 | 16384 | 767.8 | 1837.3 | 125.1 | 89.8 |

现有 torch graph 从约 490–529 μs 增至 1793–1837 μs；FlashInfer graph 仍约 76–91 μs。该结果支持设计中替换“按容量展开 KV”的方向。本脚本的 FlashInfer indices buffer 按实际页数分配，尚未衡量生产 graph 按最大页数预留元数据的预算。

变长 batch 8（`[4096,3584,3072,2560]` 重复两次）的含 prepare 耗时：

| reference eager | torch graph | FlashInfer eager | FlashInfer graph |
|---:|---:|---:|---:|
| 753.9 | 520.9 | 120.8 | 81.8 |

FlashInfer graph 相对 torch graph 为 **6.37×**。本次只在 capture 前设置该分布，没有验证同一张图从等长切换为变长。

当时由此得出的结论：继续推进 paged decode；把每步的 planning 放在各层共用的 prepare 阶段；小 batch 要单独实测；这些数字还不足以预测完整模型的加速比。当时还认为不足以打开默认的自动选择；后来 #51 的做法是 `auto` 在合适的配置下直接选 FlashInfer，依据是它自己的整模型和 HTTP 测试，不是这份微基准。

## 正确性与覆盖范围

48/48 个配置通过。每个配置对照独立 FP32 grouped-einsum oracle，使用 `atol=rtol=1e-2`；初始输入的最大绝对误差为 0.009344（所有模式合计）。除初始和计时后的输出外，还修改 Q，分别检查计算 replay 和含 prepare 的路径；再写入不同的 K/V，检查输出和整个 KV cache，要求写入精确、未写区域精确不变。

覆盖 36 个主矩阵配置（batch 1/8/32 × 实际长度 32/256/4096 × 四模式），8 个容量配置（batch 8，实际长度 256/4096，`max_context=16384`），4 个变长配置（batch 8，长度 `[4096,3584,3072,2560]` 重复两次，`max_context=4096`）。

本次长度都为整页；未验证部分页、padding lane、真实跨页增长、capture 后改变长度或页数、split 调度边界、prefill、overlap、apply/撤回、完整模型及 SM100。它是接入之前的性能证据，不是对引擎里实现的验收。

## 复现

本地 ossrun 指定 source root 为本分支 worktree，发布时检查分支名；执行命令固定 release ID。远程运行的是该 release 内的源码，既不依赖远程 git 当前分支，也不会自动取得发布后本地的新修改。

脚本要先从提交 `03579e3` 取出来（见文首）。远程命令使用独立依赖目录（FlashInfer 0.6.14 单独装，不是仓库 `gpu` extra 里的 0.7.2），完整执行脚本保存在本地 `.ossrun-artifacts/attn-matrix-command.sh`：

```sh
export ATTN_DEPS=/checkpoint/binary/rsinfer_cuda_graph/attn-bench-deps/flashinfer-0.6.14
export PATH="$ATTN_DEPS/bin:$PATH"
export FLASHINFER_WORKSPACE_BASE=/checkpoint/binary/rsinfer_cuda_graph/attn-bench-cache
export MAX_JOBS=4
bench() {
  bash oss_run_rsinfer_cuda_graph/scripts/ossrun/gpu.sh exec \
    env PYTHONPATH="$ATTN_DEPS:$PWD/engine:$PWD" python -m tests.gpu.bench_attn "$@"
}
bench --batches 1 8 32 --lengths 32 256 4096 --max-context 4096 --iterations 20 --repeats 7 --warmup 5
bench --batches 8 --lengths 256 4096 --max-context 16384 --iterations 20 --repeats 7 --warmup 5
bench --batches 8 --lengths 4096 --varlen --max-context 4096 --iterations 20 --repeats 7 --warmup 5
```

各次输出含环境、源 release、脚本 hash、每组计时、正确性和 PyTorch allocated/reserved 数据。显存数据包含基准输入、oracle 输出、两张共享池 graph 和库 workspace，不作为生产单图预算。

## 日志回收与本地验证

测量进程在 2026-11-11 04:03:14 UTC 写出最终 `RSINFER_ATTN_MATRIX_OK`。ossrun 的原命令完成状态回写滞后；未把其 `running` 状态当作成功证据。通过只读 probe `cmd_20261111T041319Z_probe_attn-fetch-probe_81ae330f` 取回三个完整 JSON，probe 为 `done / exit 0` 且匹配 `ATTN_FETCH_OK`；另行核对 36+8+4 条记录全部正确、release 和脚本 SHA256 一致。远程诊断确认没有 benchmark 进程、GPU 空闲。诊断期间另一个 GPU smoke 在 04:08:49 UTC 才开始，晚于矩阵完成，没有与测量重叠。

本地原始证据保存在 worktree 的 `.ossrun-artifacts/`（忽略，不提交生成物）：

- `attn-fetch-probe.log`：完整结果回收日志，三个 `ATTN_RESULT` 和最终成功标记。
- `attn-bench-main.json`、`attn-bench-capacity.json`、`attn-bench-varlen.json`：从日志提取并校验的结构化结果。
- `attn-matrix-command.sh`、`attn-install-command.sh`、`attn-extra-deps-command.sh`：执行和依赖安装记录。
- `attn-probe.log`、`attn-extra-deps.log`、`attn-readonly-diag.log`：环境、依赖和任务诊断。

当时带着脚本的提交在本地过了 `uv run pyright`（0 错误）和 `uv run pytest tests/units/kernels -q`（24 passed）；macOS 上的全量 `pytest -n 8` 没有跑完（481 passed、73 failed 后人工中止，日志里有只读快照 rename 的 PermissionError 和超时）。脚本去掉以后，本 PR 不含代码，这些记录只说明当时的情况。
