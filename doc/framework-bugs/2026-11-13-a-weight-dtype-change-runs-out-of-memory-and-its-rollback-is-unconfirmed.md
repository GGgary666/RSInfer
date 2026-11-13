# 把 `model.dtype` 从 bf16 热改成 f32，重读权重时 CUDA OOM，撤回没得到确认，引擎要求恢复

- 日期：2026-11-13
- 报告人：Claude Opus 5.5（Claude Code），在 4090 D 上验证 #59（前缀缓存）时碰到，见 [../rsi/2026-11-13-prefix-cache-on-a-live-engine.md](../rsi/2026-11-13-prefix-cache-on-a-live-engine.md)
- 状态：未修
- 引擎：
  - RTX 4090 D（24 GB）。Qwen3.8-27B 的真实结构截到 4 层，随机权重的 checkpoint：`deploy/agxinfer.py checkpoint --layers 4 --dtype bf16 --seed 1`，权重 7.57 GiB。
  - 起法：`deploy/agxinfer.py serve <ckpt> --tokenizer <Qwen3.6-27B> --cuda-graph auto --overlap --max-rows 8 --max-context 4096 --token-budget 2048`。代码是 master `80ee578`，之后热改上了 #59 的树。
  - 变更前身份 `200ef73d5d49b01c`，引擎是空的。holder 给状态分了 12.41 GiB（GDN 656 槽，KV 107,020 页）。
- 变更：

  ```sh
  python -m core.control.client --control <地址> apply --config '{"model.dtype": "f32", "state.gdn.conv_dtype": "f32"}'
  ```

  `--dry-run` 的计划：worker 重载 `core.state.weights`、`models.qwen38`、`layers.attn`、`layers.gdn`、`state.gdn_slots`，不要求重启。
- 现象：`status: failed`，`recovery_required: true`。耗时：check+plan 1.10 秒、prepare 0.19 秒、pause 0.011 秒、abort 0.32 秒。诊断（控制面 `events.sqlite` 的 `changes.diagnostics_json`）：

  ```text
  models.qwen38 | load | unit.not_active | is FAILED after reload: OutOfMemoryError: CUDA out of memory.
    Tried to allocate 4.74 GiB. GPU 0 has a total capacity of 23.52 GiB of which 3.62 GiB is free. Including
    non-PyTorch memory, this process has 19.89 GiB memory in use. Of the allocated memory 17.39 GiB is
    allocated by PyTorch, and 2.07 GiB is reserved by PyTorch but unallocated.
  core.plan / core.worker / executors.pytorch | load | unit.not_active | is PENDING after reload, waiting for model@1 / executor@1 / plan@1
  | rollback | unit.rolled_back | reload rolled back; the previous units are serving
  | rollback | escalate.restart_process | units did not come back: ['core.plan', 'core.worker', 'executors.pytorch', 'models.qwen38']
  host-a/worker/0 | abort | change.abort_unacknowledged | rollback not confirmed: ... (hint: restart the process)
  | abort | change.recovery_required | rollback or process stop is unconfirmed; restart the engine
  ```

  - 4.74 GiB 正好是 f32 的词表矩阵（248,320 × 5,120 × 4 字节）。旧的 bf16 权重和状态池都还在：单元运行时留着旧实例，备着撤回。
  - 之后 `status` 里 worker 的单元全是 ACTIVE，身份还是旧的。这时没再发请求试，因为要求恢复以后 `apply` 都被拒。
- 期望：
  - 放不下就失败，这没问题：要同时放得下新旧两份权重，24 GB 的卡上放不下。
  - 但撤回应当是干净的，像 [2026-11-10-gdn-ssm-dtype-rebuild-oom.md](2026-11-10-gdn-ssm-dtype-rebuild-oom.md) 那次一样：`aborted`，旧版本照常服务。而不是一面说"已撤回，旧单元在服务"，一面说"单元没回来"，最后要求恢复。
  - 两条诊断互相矛盾，说明撤回在确认哪些单元回来了这一步上判断错了，或者真有单元没回来。没查是哪一种。
- 复现：同样的引擎，同样的 `apply --config`。在一张放不下两份权重的卡上应当每次都复现；没有写成测试。
- 绕过：停掉引擎，用 `--dtype f32 --conv-dtype f32` 冷启动，接着做别的验证。之后没有再试这个热改。
