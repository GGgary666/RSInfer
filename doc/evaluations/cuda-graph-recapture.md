# CUDA Graph 配置变更：重录与复用验证（2026-11-10）

在真实 SM90-80GB 上，通过运行中的控制面 `apply` 验证了 6 组代表性配置：**3 组重新录制，3 组保留原图，全部符合预期**。每次变更后 HTTP 生成文本都与变更前一致，没有重启进程；普通 decode 不会额外触发录制。

## 方法

用例为 [test_config_changes_recapture_only_graph_dependencies](../../tests/gpu/test_cuda_graph_apply.py)。同一个 TP1 小 Qwen 引擎按顺序执行全部配置，使用随机 FP32 checkpoint、真实 tokenizer 和 HTTP 请求，`plan.cuda_graph=required`，初始 bucket 为 1/2/4。

每组执行 `apply --dry-run`、正式 `apply --config`，检查提交成功，并通过两类证据区分重录与复用：

1. 控制面 introspect 中 `core.plan` 的实例 generation。
2. 测试源码快照在每次 `recorder.capture()` 成功退出后打印专用标记，精确统计实际 capture 完成次数。该探针只注入临时测试快照，不修改生产代码；不以 plan 重载或日志出现一次代替 capture 计数。

还检查有效 bucket 集合和构建记录、没有进程重启、HTTP 生成文本与基线一致，并在请求后再次检查 generation/capture 次数不变。初始启动实际完成 3 次 capture。

## 结果

| 变更 | 预期 | Plan generation | 新增 capture 次数 | 变更后的 bucket |
|---|---|---|---:|---|
| `sched.overlap`: false → true | 不重录 | 1 → 1 | **0** | 1/2/4 |
| `sched.max_batch`: 4 → 2 | 不重录 | 1 → 1 | **0** | 1/2/4 |
| `plan.graph_buckets`: 默认生成 1/2/4 → 显式 `[1,4]` | 重录 | 1 → 2 | **2** | 1/4 |
| 再次提交 `plan.graph_buckets=[1,4]` | 不重录 | 2 → 2 | **0** | 1/4 |
| `attn.prefill_backend`: 空值 → `reference` | 重录 | 2 → 3 | **2** | 1/4 |
| `state.gdn.slots`: 0（自动容量 8）→ 显式 12 | 重录 | 3 → 4 | **2** | 1/4 |

这几组分别证明：

- live 调度项和只影响调度上限的 startup 项可以保留原图；`sched.max_batch` 不等同于 worker 的 `batch.max_rows`。
- 配置值不变时不会重复录制。
- bucket 修改会重新录制完整的新集合，当前不复用未变化的 bucket 1/4。
- 只改 prefill backend 也会重录 decode 图：当前 backend 单元共用 prefill/decode，重载沿依赖传播到 plan。这是现有实现的保守粒度。
- 状态容量变化可以在空引擎暂停期间原地重载并重录，**不必然要求重启进程**；此前部署说明中的笼统“状态单元重启规则”已更正。

## 同批回归

同一远程命令还复验：

- kernel 源码 apply → 输出确实改变；非法 bucket 配置撤回后仍可服务；恢复源码后输出回到基线，并与冷启动一致。overlap 开、关各一次。
- 同地址权重更新：保持 binding generation 和原 GraphPlan 对象，输出反映新权重并与 eager 对齐；这是 worker 内的权重 fill 验证，不是经控制面的 refit action 测试。还覆盖 stale guard 与 kernel 重载。

最终 **4 passed、5 deselected，90.12 秒**。其中新增的一个测试包含上表 6 次配置变更；不是 6 个独立 pytest case。无 skip/xfail。

## 环境与复现

- NVIDIA SM90-80GB / SM90，GPU 1，Python 3.12.7，PyTorch 2.11.0+cu130；通过 OSS runner 复用已安装环境。
- release：`20261110T110012Z_e6c6971-dirty_997e6787`。
- command：`cmd_20261110T110035Z_graph-config-recapture_76d8fc7e`；最终 done / exit 0 / `AGXINFER_GPU_TEST_OK`。
- 本地原始日志：`.ossrun-artifacts/graph-config-recapture.log`，逐项记录以 `GRAPH_CONFIG_CASE` 开头。初始 run.sh 等待进程因 OSS 读取异常退出，随后用原 CID 的 status/tail 确认最终结果，没有重复提交。

```sh
bash oss_run_agxinfer_cuda_graph/scripts/ossrun/run.sh \
  --release 20261110T110012Z_e6c6971-dirty_997e6787 \
  --inline 'bash oss_run_agxinfer_cuda_graph/scripts/ossrun/gpu.sh test tests/gpu/test_cuda_graph_apply.py tests/gpu/test_cuda_graph.py -k "config_changes or hf_guard or kernel_apply" -x' \
  --name graph-config-recapture --expect '^AGXINFER_GPU_TEST_OK$' \
  --max-wait-s 1800 --log-out .ossrun-artifacts/graph-config-recapture.log
```

本地 pyright：0 errors；CUDA Graph CPU 专项和 deps 规划测试：36 passed。此次只增加测试、共用测试辅助及说明，没有修改引擎实现，也未逐项验证所有配置或并发请求中的变更行为。
