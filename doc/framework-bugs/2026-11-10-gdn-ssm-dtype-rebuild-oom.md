# 在 SM100 上把 `state.gdn.ssm_dtype` 从 f32 改成 bf16，原地重建 GDN 状态池时 CUDA OOM，变更被撤回

- 日期：2026-11-10
- 报告人：Devin（n1 上做热更新耗时对比时碰到，是其中的 B7 这个 case）
- 状态：已修（修复和本报告在同一个 PR 里，见文末「修复」）
- 引擎：
  - 机器：n1（`gpu-node-n1`），SM100 180 GB 单卡（GPU 1），torch 2.11.0+cu130；
  - 代码：master `efce690`；
  - 命令：

    ```sh
    CUDA_VISIBLE_DEVICES=1 python deploy/qwen38.py serve --model /models/Qwen3.8-27B --source <工作区> \
        --run <run> --port 8001 --max-rows 64 --max-context 4096 --token-budget 2048 --mem-fraction 0.85 \
        --page-size 16 --timeout 3600
    ```

    其余用默认值：`model.dtype` bf16，`state.gdn.ssm_dtype` auto（checkpoint 的 `mamba_ssm_dtype` 是 float32），`state.gdn.conv_dtype` bf16。
  - 变更前身份 `bd82be18ca750147`，引擎是空的（没有请求）。
- 变更：

  ```sh
  python -m core.control.client --control <地址> apply --config '{"state.gdn.ssm_dtype": "bf16"}'
  ```

  只改这一个配置键，不改代码。`--dry-run` 给出的计划是 worker 重载 `models.qwen38` 和 `state.gdn_slots`，不要求重启。
- 现象：
  - `status: aborted`。耗时：check+plan 0.79 秒、prepare 0.17 秒、drain 0.006 秒、pause 0.014 秒、abort 2.90 秒。
  - 诊断：

    ```text
    models.qwen38 is FAILED after reload: OutOfMemoryError: CUDA out of memory. Tried to allocate 46.27 GiB.
      GPU 0 has a total capacity of 178.35 GiB of which 25.87 GiB is free. Including non-PyTorch memory, this
      process has 152.47 GiB memory in use. Of the allocated memory 105.48 GiB is allocated by PyTorch, and
      46.24 GiB is reserved by PyTorch but unallocated. ... at .../engine/platforms/torch_eager/__init__.py:129
    core.plan / executors.pytorch / core.worker: is PENDING after reload, waiting for model@1 / plan@1 / executor@1
    reload rolled back; the previous units are serving
    ```

  - worker 日志里 holder 的三条记录：

    ```text
    core.state.holder: 101.36 GiB for states: gdn.slots 334 (47.89 GiB), kv.paged 54753 (53.47 GiB)   # 启动
    core.state.holder: 101.43 GiB for states: gdn.slots 657 (48.00 GiB), kv.paged 54709 (53.43 GiB)   # 这次变更
    core.state.holder: 101.36 GiB for states: gdn.slots 334 (47.89 GiB), kv.paged 54753 (53.47 GiB)   # 回滚
    ```

  - 回滚本身是对的：旧单元继续服务，请求的输出和变更前相同；再 apply 把这个键改回 `auto` 也能提交。
  - 复现了 2 次，2 次都一样。
- 期望：
  - 引擎是空的时候，会重建状态的配置变更应该原地重建（`doc/specs/empty-apply.md` §3：#35 那条"要求重启引擎"去掉了，"引擎空了，状态单元直接原地重建"）。
  - 同样的配置冷启动能放下：`deploy/qwen38.py serve ... --ssm-dtype bf16` 能正常起来，holder 给 `gdn.slots 657 (48.00 GiB), kv.paged 54709 (53.43 GiB)`，输出和 f32 时相同。所以显存总量够，是原地重建的做法有问题。
- 原因（分析，已用下面"验证"一节确认）：
  - 换成 bf16 以后，每个槽的字节数减半，holder 在差不多的预算里给 GDN 分了 657 个槽（原来 334 个）。新的 temporal 张量是 48 × 658 × 48 × 128 × 128 × 2 字节 = 46.27 GiB，比旧的大一点。
  - 旧池子的缓冲区释放后，显存还在 PyTorch 的缓存分配器里（"46.24 GiB reserved by PyTorch but unallocated"），没有还给驱动：
    - `TorchEagerPlatform.free` 只是丢掉张量引用：`engine/platforms/torch_eager/__init__.py` 第 133–138 行；
    - `MambaPool.declare` 先 `self.free()`，再按新形状 `platform.alloc`：`engine/state/gdn_slots/unit.py` 第 64–78 行；
    - `alloc` 用 `torch.zeros` 一次申请一整块：同一文件第 129 行。
  - 缓存里最大的空闲块放不下这块稍大的新张量。分配器只好向驱动再要一整块 46.27 GiB，可驱动那边只剩 25.87 GiB，于是 OOM。缓存里的空闲加上驱动的空闲一共约 72 GiB，总量其实够。
  - 再细一层（修的时候确认的）：分配器申请失败时，会先把整块空着的缓存还给驱动再重试，所以光是有缓存还不至于 OOM。出问题的是申请的顺序：
    - `declare` 先申请 conv。新 conv（48 × 658 × 10240 × 3 × 2 字节 = 1.81 GiB）比旧 conv 那块（0.92 GiB）大，按 best-fit 从旧 temporal 那一大块里切了出来；
    - 那一大块从此算"在用"，还不回去；
    - 接着申请 temporal，缓存里没有放得下的，驱动那边又不够。
  - 新池子只要比旧池子大一点就会碰到这个问题。改 `holder.*`、`state.kv.pages`、`state.kv.dtype`（KV 存储的 `declare` 也是先 free 再 alloc，`engine/state/kv_paged/unit.py` 第 84–85 行）也可能碰到，这几个没有单独试。
- 验证：
  - 同样的引擎，worker 进程带 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 启动（节点代理会把这个变量传给引擎进程，见 `engine/core/nodeagent/agent.py` 的 `child_env`），再做同样的 apply：`committed`，execute 0.56 秒，holder 变成 657 个槽，输出相同。
  - CPU 上的测试碰不到这个问题：`tests/units/executors/test_config.py` 的 `test_a_worker_rebuilt_for_new_state_config_generates_as_one_started_with_it` 只在 CPU 上测了 `state.kv.dtype`、`state.gdn.conv_dtype`、`state.kv.pages`，CPU 上没有 CUDA 缓存分配器。
- 复现：
  1. 在 SM100 上用上面的命令起引擎（完整 27B，默认显存比例 0.85）；
  2. `apply --config '{"state.gdn.ssm_dtype": "bf16"}'`，结果是 `aborted`，诊断如上。
  
  能写成 `tests/gpu/` 下的测试：起引擎，apply 这个键，断言 `committed`，输出和冷启动 `--ssm-dtype bf16` 的相同。
- 修法的方向（写报告时考虑过的）：
  - 重建状态池时，先把缓存还给驱动再申请新的。比如状态单元在 `declare` 里 free 之后调 `torch.cuda.empty_cache()`；或者 `platforms.torch_eager` 的 `alloc` 碰到 OOM 时先 `empty_cache` 再试一次。后者属于框架本体（`engine/framework.toml` 里有 `platforms.torch_eager`）。
  - 或者 worker 进程默认用 `expandable_segments:True`，从根上避免这种碎片。可以放在部署那一层，比如 `deploy/qwen38.py` 给 worker 角色设的环境变量。
- 另外观察到（没有单独确认会不会出问题）：
  - 这次变更里 holder 把两个池子都重新分了：`kv.paged` 从 54753 页变成 54709 页。但计划只重建 GDN 存储，KV 存储不重载，还是 54753 页。
  - 所以如果这次提交成功了，两个池子加起来是 48.00 + 53.47 = 101.47 GiB，比 holder 算的预算 101.43 GiB 多 0.04 GiB。
- 绕过：
  - 冷启动时直接带 `--ssm-dtype bf16`，或者给 worker 设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后再 apply。
  - 热更新耗时对比里，B7 按"失败、已撤回"记录，没有换成绕过后的结果。
- 修复（2026-11-10）：
  - 修在单元一侧，不改框架本体，所以可以热更新上去：
    - `state.gdn_slots`：`declare` 在 `self.free()` 之后、申请之前调 `torch.cuda.empty_cache()`；
    - `state.kv_paged`：`declare` 先校验全部声明，把要换的层都 free，并丢掉它们的 `k_buffer` / `v_buffer` 引用，清一次缓存，再统一申请。
    
    两个都改，是因为同一次变更两个存储都重建时（比如改 `holder.mem_fraction`、`state.kv.page_size`），模型先给 KV 存储 `declare`，KV 的申请也可能切进 GDN 存储刚释放的大块。
  - 在 CPU 上、CUDA 没初始化时，`torch.cuda.empty_cache()` 什么都不做。
  - 测试：`tests/gpu/test_state_rebuild.py`。按 GPU 的空闲显存定大小：旧池约占 55%，新池是 bf16、槽数翻倍再多几个，直接复现切块。
    - 不修时 OOM（"Tried to allocate 97.78 GiB … 93.94 GiB is reserved by PyTorch but unallocated"）；
    - 修了以后通过。
  - n1 上的端到端验证，每次 apply 后输出都和变更前相同：
    - 用修过的代码冷启动后：B7 改成 bf16 再改回 `auto`，`holder.mem_fraction` 0.85 → 0.80 再改回，四次都 `committed`；
    - 一台没修的引擎：先 B7，`aborted`，同样的 OOM（第 3 次复现）；再把修复热更新上去（重载 `state.gdn_slots`、`state.kv_paged`），`committed`；之后 B7 `committed`。
