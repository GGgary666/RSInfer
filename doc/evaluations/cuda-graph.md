# TP1 CUDA Graph 专项验证（2026-11-10）

首版实现见 [设计与实现方案](../specs/cuda-graph.md)。按本次要求，优先验证 CUDA Graph 用例，不重复扩大到完整模型或全量 GPU 回归。

## 环境与复现

- 分支 `gw/dev/cuda-graph`，OSS project `agxinfer_cuda_graph`。
- 主机 `sm90-node-0`，NVIDIA SM90-80GB、SM90、约 80 GiB，使用 GPU 1；不是 SM100。
- Python 3.12.7，锁定 PyTorch 2.11.0+cu130 / CUDA 13.0，transformers 5.22.1。依赖通过local package mirror按 uv.lock 哈希校验安装。
- runner 及本地使用说明在 `oss_run_agxinfer_cuda_graph/action.md`，属于本机工具，不提交到源码仓库。发布后始终固定 `--release`，避免其他 session 更新 app 链接影响结果。

```sh
bash oss_run_agxinfer_cuda_graph/scripts/ossrun/run.sh --release <RID> \
  --inline 'bash oss_run_agxinfer_cuda_graph/scripts/ossrun/gpu.sh test tests/gpu/test_cuda_graph.py tests/gpu/test_cuda_graph_apply.py -x' \
  --name graph-focused --expect '^AGXINFER_GPU_TEST_OK$' \
  --max-wait-s 1800 --log-out .ossrun-artifacts/graph-focused.log
```

`--max-wait-s` 只限制本地等待；没有给 pytest 添加总运行时限。成功同时要求 done、exit 0 和成功标记。

## GPU 结果

源码 release `20261110T101318Z_4a5f556-dirty_2e26f17f`：

| 范围 | 结果 | OSS command |
|---|---|---|
| `test_cuda_graph.py` + `test_cuda_graph_apply.py` | **8 passed，63.34 秒** | `cmd_20261110T101342Z_graph-focused-v2_665bd9c1` |
| `test_qwen38_partial.py -k True`，仅 graph 的 fp32/bf16 两项 | **2 passed，2 deselected，37.77 秒** | `cmd_20261110T101548Z_graph-real-shapes_ef80fafe` |

后续只增强 refit 断言、同步文档；release `20261110T101647Z_4a5f556-dirty_c6b764e6` 的 `test_cuda_graph.py -k hf_guard`：**1 passed，5 deselected，4.87 秒**，command `cmd_20261110T101705Z_graph-refit_9b29cf00`。本地等待曾因 OSS 读取错误退出；随后 status 确认 done/exit 0，完整日志匹配成功标记，没有重复提交。

专项覆盖：

- 平台复用 capture stream、非法 capture 后能重新捕获、close 后拒绝 replay。
- 完整 forward 的 logits 与 KV/GDN 状态对照同配置 eager；token、长度、行映射变化，bucket 1/2/4 切换、padding、prefill/mixed 交替、before/after StateCopy。
- 小 Qwen 的 HF logits/token 对照；换成 dummy 权重后绑定不变、同一张 graph 输出改变并与新权重 eager 一致。
- 绑定代数变化在写状态前拒绝；kernel 重载使旧图失效；不可捕获 backend 的 auto/required 行为。
- 第二个 bucket 注入构建失败，未发布部分 graph；清理后 allocated 显存回到基线 + 1 MiB 内。
- overlap 深度 1/2 输出一致；4 次重录后，后 3 次 allocated 显存波动不超过 1 MiB。
- 真实 HTTP 引擎执行 `check → apply --dry-run → apply`：修改 linear kernel 确认输出变化；非法 bucket 配置失败后恢复服务；恢复源码后输出回到基线并与冷启动一致。overlap 开、关各一次。

真实层尺寸测试使用 Qwen3.8-27B 的前 4 层随机权重（BF16 checkpoint 7.57 GiB）、bucket 1/2/4/8、最大上下文 256。fp32 下 3 个 prompt 各 12 个 greedy token 与 HF 全部一致；prefill logits 通过原有 1e-3 容差。BF16 prefill cosine 为 0.99977 / 0.99971 / 0.99981；首次 greedy 分歧在第 0 / 2 / 2 个 token。BF16 测试不声称 token 恒等，也不将 prefill 指标当作逐步 decode logits 误差。

构建日志的累计峰值增量：fp32 383216640 字节（约 365.5 MiB），bf16 367796224 字节（约 350.8 MiB），均低于 1 GiB 预算。该数字包含构建临时张量和 workspace，不是单独 graph 常驻大小；不是 TPOT 或吞吐基准。

本地原始日志保存在 `.ossrun-artifacts/graph-focused-v2.log`、`graph-real-shapes.log` 和 `graph-refit.log`；生成物不进 Git。

## 本地检查与限制

- Python 3.12 独立 CPU 环境，`pyright`：0 errors。
- executor/state/kernel/contracts/deps/platform 定向回归：125 passed。
- 加载顺序修正后的 unit_runtime/platform/deps 回归：98 passed（与上一组有重叠，不相加）。
- formal 全流程（模型、mutant、witness、轨迹回放）通过；Python 回放 59 passed。使用校验过的 TLC 1.7.4 和本地 Java 21。
- 较早的 macOS 全量 `pytest -n 8` 在 22 分钟后主动中断，452 passed、70 failed，**全量未通过**。日志可见 snapshot 物化 rename 的 `PermissionError`、清理子进程的 `killpg` 权限错误、SHM flock 不支持和 TCP 时序失败。未逐项证明全部 70 项都与本补丁无关；后续按要求聚焦专项，不将此结果表述为全量通过。

首次 GPU 专项发现 kernel 注册晚于 plan 构建，已修复并记录在 [加载顺序报告](../framework-bugs/2026-11-10-capture-before-registration.md)。框架改动限于 CUDA 平台入口、warm-up/close 原语及该加载顺序修正；不修改事务协议。

尚未验证完整 27B 真权重/SM100、长上下文和大 bucket 的性能；未接入优化 attention/GDN 库，未承诺加速倍数。
