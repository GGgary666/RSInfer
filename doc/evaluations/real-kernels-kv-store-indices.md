# 真 kernel 第三批：KV 写入和 KV 槽列表各合成一个 kernel（2026-11-11，RTX 4090 D，4 层）

接着 [real-kernels-fused.md](real-kernels-fused.md)。那一批之后，一个回放的 decode 步里和 SGLang 还不一样的两处是：KV 写入是两次 `index_put`，以及每步拼 KV 槽列表用了十来个 torch 操作。Gemini 在 SM100 上估这两项合计约 0.3 ms（[real-kernels-sm100.md](real-kernels-sm100.md)）。

这一批的代码是 Gemini 4 Carbon 写的（分支 `exp/sm100-kv-store-indices` 的前四个提交），写到一半额度用完；我（Claude Fable 5.2）接手做完：把它在这台机器上跑通、修了几处、补了测试和文档。**SM100 上的效果、SM90 以上才有的 PDL 路径，我这里都跑不了，没有验证。**

## 结论

1. **这两处都换成一个 kernel 了，结果和参照实现逐位相同。** 在引擎里用真实请求查过，也有 GPU 测试。
2. **一个回放的 decode 步现在是 74 次 launch**（上一批 85 次，SGLang 82 次），都是在跑着的引擎里用 profiler 量的。attention 层每层 12 个 kernel，和 SGLang 相同。
3. **这张卡上时间没有可见的变化**：设备上的 kernel 时间 6.141 → 6.121 ms，一步仍然压在 GEMM 上（5.92 ms）。这一批是为 SM100 做的。
4. **新单元 `kernels.cuda` 在加载时编译 C++，约一分钟。** 冷启动和"热改时新增这个单元"都没问题；改它的 C++ 源码再热改，要先在引擎外编译（见"热改"）。

## 做了什么

| | 之前 | 现在 |
|---|---|---|
| `kv.store@1` | 参照实现：K、V 各一次 `index_put`，2 个 kernel | `cuda.kv.store@1`：1 个 kernel。照 SGLang 的 `store_kvcache`（`kernels/jit/csrc/elementwise/kvcache.cuh`）写：一行拆给几个 warp，16 字节一组地读写；SM90 以上带 PDL |
| 每步拼 KV 槽列表、取每行要写的槽 | `layers.attn` 里用 torch 拼（`searchsorted`、gather、拷贝） | 新 definition `attn.kv_indices@1`。`triton.attn.kv_indices@1` 是 SGLang 的 `create_flashinfer_kv_indices_triton`，多了一个可选的输出：每行最后一个槽（decode 步要写的位置） |

- **`kernels.cuda`（新单元）**：SGLang 的这个 kernel 是 C++ 写的，靠它自己的 JIT 头文件编译，搬不过来。这里照它的结构重写，只依赖 PyTorch 的扩展头文件，用 `torch.utils.cpp_extension.load` 编译。
  - 源码放在 Python 模块 `kv_store_cu.py` 的一个字符串里：这样改 C++ 也在 `core.deps` 的 import 图里，会被计划成重载 `kernels.cuda`。
  - 编译产物按源码的摘要命名。改了源码就是另一个模块，和旧的并存（原生代码卸不掉）；没改就从缓存加载。
  - kernel 不接的输入（K/V 和池的 dtype 不同、一行不是整数个 16 字节、没对齐）在启动 kernel 之前就返回，退回参照实现的写法。
  - 没有 CUDA 设备的进程里什么都不编译、不登记，CI 不受影响。
- **`attn.kv_indices@1`**：参照实现用 torch 写，不在 host 上读任何值，所以可以被录进 graph；graph 桶的固定 buffer 里列表之后的部分保持原样。
- **`layers.attn`**：`prepare` 改成调 `attn.kv_indices@1`；它和 `kv.store@1` 一样算"不属于 attention backend 自己"的 definition，点名 `flashinfer` 时按 `auto` 选。

我接手后改的：

- `kernels/cuda/ops.py` 里临时文件名用了进程号，P7 的扫描测试（`test_location_transparency`）因此失败；换成 `tempfile`。
- `layers/attn/paged.py` 里原来拼列表的函数没人调用了，删掉；两个测试改成对照参照实现。
- 新的 KV 写入测试断言写错了（只有 1 个 token 时切片本来就是连续的），3 个参数全部失败；重写，并补上说明里提到却没测的两条回退路径。
- `kernels.cuda`：没编译过的源码在编译前打一行提示；加了预编译入口（`python -m kernels.cuda.ops`）；补了出处说明和 `LICENSE.sglang`。

## 数字

都是在跑着的引擎里量的（临时把 profiler 包进 executor，8 个并发请求，取 20 个 8 行的 decode 步；做法见 skill「在引擎里做检查」）。

| 回放的 decode 步，8 行 | 上一批 | 这一批 | SGLang（它自己的 trace） |
|---|---:|---:|---:|
| launch 次数 | 85 | 74 | 82 |
| 设备上 kernel 时间 | 6.141 ms | 6.121 ms | 6.22 ms |
| 其中 GEMM | 5.923 ms | 5.924 ms | 5.98 ms |
| KV 写入 | 2 次，0.0054 ms | 1 次，0.0013 ms | 1 次 |
| 拼 KV 槽列表的 kernel | torch 的几个操作 | 1 次，0.0013 ms | 1 次 |

- 少的 11 次里，1 次是 KV 写入，10 次是每步的准备。
- 没有重跑 HTTP 压测：这张卡上一步 6 ms 多里 GEMM 占 5.9 ms，少掉的 0.02 ms 量不出来。

## 在引擎里查的

临时把检查写进 `layers/attn/paged.py`，apply，发请求（8 个并发各 80 个 token，外加一个 900 词的 prompt），graph 开着和关着各一遍：

| 检查 | 次数 | 结果 |
|---|---:|---|
| 槽列表和每行要写的槽，对照参照实现（graph 桶的回放步） | 119 | 全部相同 |
| 同上（eager 的 decode 步） | 80 | 全部相同 |
| 同上（prefill 步，只比列表） | 5 | 全部相同 |
| KV 写入以后，池里那些槽的内容等于这一步的 K、V（eager decode） | 80 | 全部相同 |
| 同上（prefill，多个 token） | 5 | 全部相同 |

## 热改

这一批是在一个一直跑着的引擎上做的（13 次 apply 提交、1 次撤回、3 次 dry-run），过程记在 [../rsi/2026-11-11-kv-store-indices-on-a-live-engine.md](../rsi/2026-11-11-kv-store-indices-on-a-live-engine.md)。和 `kernels.cuda` 有关的三条：

- **把这个分支整个热改到上一批的引擎上：提交。** 新单元在准备阶段加载，55 秒的编译发生在那里，引擎没有暂停。
- **改 C++ 源码直接 apply：撤回。** 已有的单元在暂停期内重载，编译 54 秒超过了 30 秒的执行时限；引擎回了 54 秒的 503，然后照常服务旧代码。
- **先在引擎外编译，再 apply：2.6 秒提交。** 编译的 56 秒里请求全部正常。

  ```sh
  PYTHONPATH=engine uv run python -m kernels.cuda.ops   # 和 worker 同一个环境
  ```

## 测试

- `uv run pyright` 0 错误；`uv run pytest -n 8` 597 个通过。
- `tests/gpu/`（4090 D，逐个文件跑）全部通过：`test_real_kernels` 26 个，`test_qwen38_partial` 4 个，`test_cuda_graph` 9 个，`test_cuda_graph_adversarial` 3 个，`test_state_rebuild` 1 个，`test_cuda_graph_apply` 4 个，`test_qwen38_overlap` 1 个，`test_qwen38_serve` 2 个。
- 新的测试：
  - `test_cuda_kv_store_stays_within_its_definition`（bf16、fp16、fp32）：1 个、13 个、2000 个 token，int64 和 int32 的槽号，V 是 QKV 投影的 strided 切片；整个池和参照实现逐位相同。两条回退路径（dtype 不同、一行 12 字节）也是。
  - `test_triton_kv_indices_stays_within_its_definition`：跨多个 512 的块、1 个 token 的行、buffer 的尾部不动。
  - worker 栈的测试里加了断言：`auto` 选的是 `cuda.kv.store@1` 和 `triton.attn.kv_indices@1`；带着请求重载 kernel 单元时把 `kernels.cuda` 也重载两次。
  - CPU（进 CI）：`attn.kv_indices@1` 的参照实现对着手写的列表查；一行单独算和夹在别的行中间算结果相同。

## 没验证的，和没做的

- **SM90 以上的 PDL 路径没跑过。** 这张卡是 SM89，编译和运行走的都是不带 PDL 的那一支。PyTorch 的扩展加载器按当前设备选编译目标，所以 PDL 的那段代码只有在 SM90 以上的机器上才会被编译进去。
- **SM100 上的效果没量。** 这一批针对的是 Gemini 估的 0.3 ms 左右；它估的另一项（SGLang 把 `in_proj_ba` 放到第二个 stream 上并行，约 0.24 ms）没有做，那要平台提供第二个 stream。
- 没有编译器的机器上，`kernels.cuda` 加载失败，引擎起不来。这和 FlashInfer 第一次 plan 时的要求一样，只是提前到了启动时。
- prefill 仍然不走 graph。
