# 真 kernel 第三批：KV 写入和 KV 槽列表各合成一个 kernel（2026-11-11 / 11-12，RTX 4090 D 与 SM100）

接着 [real-kernels-fused.md](real-kernels-fused.md)。那一批之后，一个回放的 decode 步里和 SGLang 还不一样的两处是：KV 写入是两次 `index_put`，以及每步拼 KV 槽列表用了十来个 torch 操作。Gemini 在 SM100 上估这两项合计约 0.3 ms（[real-kernels-sm100.md](real-kernels-sm100.md)）。

这一批的代码是 Gemini 4 Carbon 写的（分支 `exp/sm100-kv-store-indices` 的前四个提交），写到一半额度用完；Claude Fable 5.2 接手在 RTX 4090 D（4 层）上跑通、修了几处、补了测试和初版文档；随后 Gemini 4 Carbon（Antigravity）在单卡 NVIDIA SM100（`sm_100`，CUDA 13.0，完整 64 层 Qwen3.8-27B）上补完 SM100 PDL 路径验证、64 层真引擎在线 C++ 热改实验、算子微基准以及对原生 SGLang 的端到端压测（2026-11-12）。

## 结论

1. **这两处都换成一个 kernel 了，结果和参照实现逐位相同。** 4090 D（SM89）和 SM100（SM100，`kUsePDL = true`）上的引擎内检查、`tests/gpu/` 全量 8 个文件 50 项测试、Linux 默认 597 项测试全部通过。
2. **4 层回放的 decode 步现在是 74 次 launch**（上一批 85 次，SGLang 82 次），attention 层每层 12 个 kernel，和 SGLang 相同；**SM100 64 层回放的 decode 步**里 Kineto 确认 `store_kv_kernel<long, true>`（带 PDL）每步执行 16 次、`create_flashinfer_kv_indices_triton` 每步执行 1 次。
3. **SM100 完整 64 层 Qwen3.8-27B 上，HTTP decode 全并发档位（1–64）每 token 间隔比第二批再降 0.20–0.41 ms**：1–32 并发吞吐从 SGLang `fi+triton` 的 92.7–94.8% 升到 **96.0–96.9%**（1 并发热态 93.4 token/s、TPOT 10.46 ms，距 SGLang 的 10.14 ms 仅差 **0.32 ms**），64 并发升到 **3,773.3 token/s（102.1%）**；2000 token 长输入 prefill 达到 SGLang `fi+triton` 的 **98.4–101.2%**（默认后端的 **100.1–103.0%**）。4090 D 上一步压在 GEMM 上（6.141 → 6.121 ms），时间没有可见变化。
4. **新单元 `kernels.cuda` 在加载时编译 C++（4090 D 约 55 秒，SM100 / CUDA 13.0 约 123 秒）。** 冷启动和"热改时新增这个单元"都落在准备阶段（默认时限 600 秒），在 64 层 147.8 GiB 真引擎上一次性提交成功；改已加载单元的 C++ 源码先在引擎外跑 `python -m kernels.cuda.ops`（期间持续打入的 155 个请求全部 200 OK），随后 `apply` 复用缓存并重录 64 层 12 个桶仅需 **6.4 秒**。

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

Claude 接手后改的：

- `kernels/cuda/ops.py` 里临时文件名用了进程号，P7 的扫描测试（`test_location_transparency`）因此失败；换成 `tempfile`。
- `layers/attn/paged.py` 里原来拼列表的函数没人调用了，删掉；两个测试改成对照参照实现。
- 新的 KV 写入测试断言写错了（只有 1 个 token 时切片本来就是连续的），3 个参数全部失败；重写，并补上说明里提到却没测的两条回退路径。
- `kernels.cuda`：没编译过的源码在编译前打一行提示；加了预编译入口（`python -m kernels.cuda.ops`）；补了出处说明和 `LICENSE.sglang`。

## RTX 4090 D（4 层）上的验证与热改

### 数字

都是在跑着的引擎里量的（临时把 profiler 包进 executor，8 个并发请求，取 20 个 8 行的 decode 步；做法见 skill「在引擎里做检查」）。

| 回放的 decode 步，8 行 | 上一批 | 这一批 | SGLang（它自己的 trace） |
|---|---:|---:|---:|
| launch 次数 | 85 | 74 | 82 |
| 设备上 kernel 时间 | 6.141 ms | 6.121 ms | 6.22 ms |
| 其中 GEMM | 5.923 ms | 5.924 ms | 5.98 ms |
| KV 写入 | 2 次，0.0054 ms | 1 次，0.0013 ms | 1 次 |
| 拼 KV 槽列表的 kernel | torch 的几个操作 | 1 次，0.0013 ms | 1 次 |

- 少的 11 次里，1 次是 KV 写入，10 次是每步的准备。
- 这张卡上没有重跑 HTTP 压测：一步 6 ms 多里 GEMM 占 5.9 ms，少掉的 0.02 ms 量不出来。

### 在引擎里查的

临时把检查写进 `layers/attn/paged.py`，apply，发请求（8 个并发各 80 个 token，外加一个 900 词的 prompt），graph 开着和关着各一遍：

| 检查 | 次数 | 结果 |
|---|---:|---|
| 槽列表和每行要写的槽，对照参照实现（graph 桶的回放步） | 119 | 全部相同 |
| 同上（eager 的 decode 步） | 80 | 全部相同 |
| 同上（prefill 步，只比列表） | 5 | 全部相同 |
| KV 写入以后，池里那些槽的内容等于这一步的 K、V（eager decode） | 80 | 全部相同 |
| 同上（prefill，多个 token） | 5 | 全部相同 |

### 热改

4090 D 上这一批是在一个一直跑着的引擎上做的（13 次 apply 提交、1 次撤回、3 次 dry-run），过程记在 [../rsi/2026-11-11-kv-store-indices-on-a-live-engine.md](../rsi/2026-11-11-kv-store-indices-on-a-live-engine.md)。和 `kernels.cuda` 有关的三条：

- **把这个分支整个热改到上一批的引擎上：提交。** 新单元在准备阶段加载，55 秒的编译发生在那里，引擎没有暂停。
- **改 C++ 源码直接 apply：撤回。** 已有的单元在暂停期内重载，编译 54 秒超过了 30 秒的执行时限；引擎回了 54 秒的 503，然后照常服务旧代码。
- **先在引擎外编译，再 apply：2.6 秒提交。** 编译的 56 秒里请求全部正常。

  ```sh
  PYTHONPATH=engine uv run python -m kernels.cuda.ops   # 和 worker 同一个环境
  ```

## SM100（`sm_100`，完整 64 层 Qwen3.8-27B）上的实测

机器和配置与 [real-kernels-sm100.md](real-kernels-sm100.md) 完全一致：单卡 NVIDIA SM100（`sm_100`，183 GiB，CUDA 13.0 `V13.0.48`，驱动 `580.105.08`），TP=1，完整 64 层 Qwen3.8-27B bf16 真权重（48 层 GDN + 16 层 full attention，SSM 状态 fp32）；服务端参数 `--cuda-graph required --overlap --max-rows 64 --max-context 4096 --token-budget 8192 --mem-fraction 0.80`（默认录齐 1、2、4、8、12、16、24、32、40、48、56、64 共 12 档 decode graph，`kept_bytes = 278,921,216`，状态池 92.44 GiB：GDN 槽 305 占 43.73 GiB，KV 页 49,879 占 48.71 GiB）。

### SM100 PDL 验证与算子微基准

- **Kineto profiler 核对（64 层，8 行 decode graph 回放步）**：每步共 944.4 个 CUDA 事件，其中 `void (anonymous namespace)::store_kv_kernel<long, true>(...)`（模板参数 `kUsePDL = true`，确认 SM100 下内联 PTX `griddepcontrol.wait` / `launch_dependents` 和 `cudaLaunchAttributeProgrammaticStreamSerialization` 已编译并走通）精确执行 **16.0 次/步**，`create_flashinfer_kv_indices_triton` 执行 **1.0 次/步**。
- **16 层 `kv.store@1` 微基准**（CUDA Event 40 次中位数；输入 `v` 取 QKV 投影真实的非连续切片 `qkv[:, 13312:].view(tokens, 4, 256)`。注：第二轮微基准里喂的是连续张量，测得参照实现 0.23–0.25 ms；换成真实 strided `v` 后参照实现为 0.323–0.347 ms）：

| token 数 | 场景 | 参照实现（2× `index_put`，16 层） | `cuda.kv.store@1`（PDL，16 层） | 加速比 / 节省 |
|---:|---|---:|---:|---:|
| 1 | decode `b=1` | 0.347 ms | 0.115 ms | 3.0x（−0.232 ms） |
| 4 | decode `b=4` | 0.323 ms | 0.115 ms | 2.8x（−0.208 ms） |
| 8 | decode `b=8` | 0.325 ms | 0.115 ms | 2.8x（−0.209 ms） |
| 16 | decode `b=16` | 0.325 ms | 0.116 ms | 2.8x（−0.209 ms） |
| 32 | decode `b=32` | 0.324 ms | 0.115 ms | 2.8x（−0.209 ms） |
| 64 | decode `b=64` | 0.325 ms | 0.115 ms | 2.8x（−0.210 ms） |
| 2000 | prefill `1×2000` | 0.344 ms | 0.116 ms | 3.0x（−0.229 ms） |
| 8000 | prefill `4×2000` | 1.120 ms | 0.172 ms | 6.5x（−0.948 ms） |

- **每步一次的 `attn.kv_indices@1` 微基准**（CUDA Event 40 次中位数）：

| 序列长度 | 行数 `b` | 参照实现（torch 算子链） | `triton.attn.kv_indices@1` | 加速比 / 节省 |
|---:|---:|---:|---:|---:|
| 129 | 1 / 8 / 32 / 64 | 0.216 / 0.213 / 0.213 / 0.218 ms | 0.029 / 0.029 / 0.028 / 0.030 ms | 7.3–7.6x（约 −0.185 ms） |
| 2048 | 1 / 8 / 32 / 64 | 0.218 / 0.215 / 0.213 / 0.214 ms | 0.029 / 0.029 / 0.028 / 0.029 ms | 7.4–7.6x（约 −0.186 ms） |

### 64 层真引擎上的 C++ 在线热改

从第二批 `048ca47` 冷启动完整 64 层引擎（47.04 秒就绪，显存 147,786 MiB，身份 `e3830a2ce3e7aebf`），在同一个跑着的引擎上连续执行三次热 `apply`：

| 步骤 | 动作计划（`worker/0`） | 耗时与过程 | 身份、显存与输出核对 |
|---|---|---|---|
| 1. 从 `048ca47` 热改到 `f543495`（清空 `TORCH_EXTENSIONS_DIR`） | 准备阶段新增 `kernels.cuda`；重载 `core.plan`、`kernels.reference`、`kernels.triton`、`layers.attn` | **131.19 秒一次性提交**。CUDA 13.0 编译 `<torch/extension.h>` 到 `sm_100` 耗时约 123 秒（比 4090 D 的 55 秒长一倍多），因新单元加载在准备阶段（时限 `change.prepare_timeout_s`），未触发 30 秒执行超时 | 身份 `7c3fa16970b5fb4e`，显存 147,806 MiB（+20 MiB），`core.plan` 切至 `cuda.kv.store@1` + `triton.attn.kv_indices@1`，贪心输出与 `048ca47` 逐字相同 |
| 2. 改 `kv_store_cu.py` 的 C++ 源码 → 引擎外预编译 → `apply` | 重载 `core.plan`、`kernels.cuda` | 引擎外 `python -m kernels.cuda.ops` 耗时 128.24 秒（`nvcc` 123.06 秒），期间每 0.6 秒打入一个 `/v1/completions` 请求，**155 / 155 全部 200 OK（0 错误）**；随后 `apply` 命中缓存 `.so`，重录 64 层 12 个桶 **6.43 秒提交**（不重读 54 GiB 权重，不重建 92.4 GiB 状态池） | 贪心输出逐字相同 |
| 3. 还原 `kv_store_cu.py` 至 `f543495` → `apply` | 重载 `core.plan`、`kernels.cuda` | **6.91 秒提交** | 身份精确回到 `7c3fa16970b5fb4e`，显存保持 **147,806 MiB**（三次热改 0 MiB 泄漏，`kept_bytes = 278,921,216` 不变），贪心输出和 HTTP 压测与全新冷启动的 `f543495`（55.0 秒就绪）一致 |

### HTTP decode（64 层，prompt 32，生成 256）

`tests/gpu/bench_serving.py` 流式贪心压测，吞吐为输出 token/s，后面是每 token 间隔（TPOT）中位数。第三批同时列出上面经过三次热改的引擎（"热改后"）和全新冷启动的引擎（"冷启动"）：

| 并发 | SGLang 默认 | SGLang `fi+triton` | agxinfer 第二批（`048ca47`） | agxinfer 第三批热改后 | agxinfer 第三批冷启动 | 第三批相对 `fi+triton` | TPOT 相比第二批 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 98.0 / 10.04 ms | 97.0 / 10.14 ms | 89.9（热态 91.5）/ 10.87（10.68）ms | 93.2（热态 93.2）/ 10.46（10.47）ms | 91.3（热态 93.4）/ 10.69（10.46）ms | **96.3%**（差距 0.32 ms） | −0.22 ~ −0.41 ms |
| 4 | 373.3 / 10.37 ms | 370.7 / 10.48 ms | 351.4 / 11.13 ms | 359.2 / 10.87 ms | 358.7 / 10.90 ms | **96.9%** | −0.23 ~ −0.26 ms |
| 8 | 714.4 / 10.85 ms | 709.7 / 10.93 ms | 669.3 / 11.44 ms | 682.5 / 11.22 ms | 682.4 / 11.35 ms | **96.2%** | −0.09 ~ −0.22 ms |
| 16 | 1,343.7 / 11.58 ms | 1,334.0 / 11.63 ms | 1,261.9 / 12.18 ms | 1,280.6 / 11.98 ms | 1,278.6 / 12.00 ms | **96.0%** | −0.18 ~ −0.20 ms |
| 32 | 2,377.1 / 12.79 ms | 2,393.7 / 12.79 ms | 2,262.6 / 13.58 ms | 2,302.8 / 13.19 ms | 2,298.0 / 13.22 ms | **96.2%**（对默认 96.9%） | −0.36 ~ −0.39 ms |
| 64 | 3,687.7 / 15.67 ms | 3,697.3 / 15.60 ms | 3,730.1 / 16.30 ms | 3,760.8 / 16.04 ms | 3,773.3 / 15.97 ms | **102.1%**（对默认 102.3%） | −0.26 ~ −0.33 ms |

- 这一组（32 token 短 prompt）第三批冷启动的首 token 时间中位数：1 / 4 / 8 / 16 / 32 / 64 并发分别为 77.3 / 74.8 / 102.5 / 129.8 / 180.4 / 202.2 ms（8 并发从第二批的 131.8 ms 降到 102.5 ms，接近 SGLang 的 99.8 ms）。
- 热改过三次的引擎与全新冷启动的引擎在全并发档位吞吐相差不到 0.4%，确认热改后没有残留开销。

### HTTP prefill（64 层，prompt 2000，生成 8）

吞吐为输入 token/s，后面是首 token 时间中位数。`tb` 是 `--token-budget`（`tb=16384` 对齐 SGLang 的 `chunked_prefill_size`）：

| 并发 | SGLang 默认 | SGLang `fi+triton` | 第二批（`tb=8192`） | 第二批（`tb=16384`） | 第三批（`tb=8192`） | 第三批（`tb=16384`） | 第三批（`tb=16384`）相对 `fi+triton` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10,914 / 106.1 ms | 11,220 / 103.1 ms | 10,894 / 103.8 ms | 10,962 / 102.7 ms | 11,059 / 102.3 ms | 11,072 / 102.2 ms | **98.7%**（对默认 101.4%） |
| 4 | 16,988 / 386.3 ms | 17,289 / 381.0 ms | 16,936 / 291.8 ms | 16,934 / 293.4 ms | 17,010 / 293.5 ms | 16,938 / 297.2 ms | **98.0%**（`tb=8k` 为 **98.4%**，对默认 100.1%） |
| 8 | 18,997 / 748.3 ms | 19,350 / 738.6 ms | 19,257 / 534.0 ms | 19,386 / 728.1 ms | 19,358 / 531.9 ms | 19,575 / 724.0 ms | **101.2%**（对默认 103.0%） |

### 进程内步级拆解与对第二轮估算的复盘

不挂 profiler（避免 CUPTI 回调拉高 `submit_ms`），直接对比第二批（`048ca47`）和第三批（`f543495`）的 64 层进程内步级计时：

- **decode 非重叠（`depth=1`，`prepare` 与 graph 回放串行在关键路径上）**：

| batch | 第二批每步 / host 提交 / 设备 | 第三批每步 / host 提交 / 设备 | 变化 |
|---:|---:|---:|---:|
| 1 | 11.48 / 0.83 / 11.35 ms | 11.30 / 0.76 / 11.15 ms | 每步 −0.18 ms，提交 −0.07 ms，设备 −0.20 ms |
| 4 | 11.86 / 0.87 / 11.61 ms | 11.83 / 0.87 / 11.61 ms | 每步 −0.03 ms |
| 8 | 12.26 / 0.91 / 11.97 ms | 12.17 / 0.82 / 11.90 ms | 每步 −0.09 ms，提交 −0.09 ms，设备 −0.07 ms |
| 16 | 13.18 / 0.95 / 12.75 ms | 13.08 / 0.88 / 12.69 ms | 每步 −0.10 ms，提交 −0.07 ms，设备 −0.06 ms |
| 32 | 14.59 / 1.06 / 13.97 ms | 14.57 / 1.00 / 13.95 ms | 每步 −0.02 ms，提交 −0.06 ms |
| 64 | 17.20 / 1.30 / 16.13 ms | 17.11 / 1.18 / 16.07 ms | 每步 −0.09 ms，提交 −0.12 ms，设备 −0.06 ms |

- **decode 紧耦合双缓冲（`depth=2`）**：`b = 1 / 4 / 8 / 16 / 32 / 64` 每步中位数为 `10.73 / 11.11 / 11.44 / 12.20 / 13.39 / 15.43 ms`（`93.2 / 359.9 / 699.0 / 1,311.4 / 2,390.5 / 4,148.5 token/s`），host 提交 `0.76–1.39 ms`，设备执行 `10.71–15.26 ms`，步间空闲 `0.003 ms`。
- **prefill 进程内（`prompt_len = 2000`）**：`1×2000` 端到端 `97.95 ms`（第二批 `98.03 ms`）、host 提交 `58.78 ms`（第二批 `59.72 ms`）；`4×2000` 端到端从 `368.35 ms` 降到 **`366.13 ms`（−2.22 ms）**，host 提交从 `83.25 ms` 降到 **`77.27 ms`（−5.98 ms）**，吞吐升至 `21,850 token/s`。

对照 [real-kernels-sm100.md](real-kernels-sm100.md) 末尾备注 #2，这组数据把账彻底算清了：

1. **为什么单算子微基准省约 0.40 ms（0.21 + 0.19），而进程内紧耦合 `depth=2` 只小幅变动？** 单测里两次 `index_put` 测出的 0.33 ms 大头是 eager 模式下的 host 调度和 kernel 启动气泡——录进 CUDA Graph 以后，这部分启动开销本来就已被图摊掉，16 层从 32 个 graph 节点缩成 16 个带 PDL 的 `store_kv_kernel` 在纯硅片上省的是约 0.02–0.05 ms；同时进程内紧耦合 `depth=2` 在上一步 10.7 ms 的图执行期间就把下一步 `prepare` 的 10 个小算子提前塞进了流里，气泡也被藏住了。
2. **为什么真实 HTTP serving 全并发（1–64）稳定快了 0.20–0.41 ms/步？** 真实服务链路经由 `frontend` → ZMQ → `core.sched` → `core.worker`：1 并发单序列必须等上一步采样完才能决定下一步请求，走的是 `depth=1` 关键路径；中高并发下多进程收发与流同步之间无法像合成死循环那样零空隙预排队列，graph 外砍掉 10 次 torch 小算子（`0.215 → 0.029 ms`）加图内减少 16 个节点，直接转化成了端到端 TPOT 的 0.20–0.41 ms 下降。
3. **剩下的 0.32 ms 在哪**：1 并发热态 HTTP TPOT 现在是 10.46 ms，对 SGLang `fi+triton` 的 10.14 ms 只差 0.32 ms，基本就是本分支按约定没动的 48 层 GDN 双 stream `in_proj_ba` 并行（估约 0.24 ms）加上跨进程 ZMQ/Python 调度开销。

### 4 层 microbench（SM100，`tests/gpu/bench_cuda_graph.py`，`max_context` 4096）

每步毫秒，括号里是 graph 的 token/s：

| batch | depth | 第二批 eager / graph | 第三批 eager / graph | 第三批 host 提交 eager / graph | 第三批设备执行 eager / graph |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 3.28 / 1.75（570.1） | 3.19 / 1.68（596.1） | 2.94 / 0.71 ms | 3.07 / 1.52 ms |
| 1 | 2 | 3.09 / 1.18（850.1） | 3.05 / **1.11（899.8，+5.8%）** | 2.95 / 0.71 ms | 3.01 / 1.10 ms |
| 4 | 1 | 3.37 / 1.86（2,147.6） | 3.33 / 1.77（2,264.8） | 3.00 / 0.75 ms | 3.15 / 1.58 ms |
| 4 | 2 | 3.19 / 1.21（3,317.7） | 3.22 / **1.15（3,470.1，+4.6%）** | 2.99 / 0.74 ms | 3.07 / 1.13 ms |
| 8 | 1 | 3.47 / 1.97（4,062.1） | 3.43 / 1.88（4,254.0） | 3.05 / 0.78 ms | 3.20 / 1.63 ms |
| 8 | 2 | 3.30 / 1.24（6,470.6） | 3.27 / **1.18（6,802.7，+5.1%）** | 3.06 / 0.76 ms | 3.14 / 1.16 ms |

- `max_context` 256 时第三批 `depth=2` 的 graph 为 1.11 / 1.15 / 1.18 ms（898.3 / 3,490.9 / 6,805.9 token/s），与 4096 一致；全部 12 组用例 graph 与 eager 的 `token_sha256` 完全一致。

## 测试

- `uv run pyright` 0 错误；`framework_changed.py 048ca47 HEAD` 为 `false`；Linux 上 `uv run pytest -n 8` 597 个通过。
- `tests/gpu/` 在 **RTX 4090 D（SM89）** 和 **NVIDIA SM100（SM100）** 两台机器上逐个文件跑均全量通过（共 8 个文件 50 项）：`test_real_kernels` 26 个，`test_qwen38_partial` 4 个，`test_cuda_graph` 9 个，`test_cuda_graph_adversarial` 3 个，`test_state_rebuild` 1 个，`test_cuda_graph_apply` 4 个，`test_qwen38_overlap` 1 个，`test_qwen38_serve` 2 个。
- 新的测试：
  - `test_cuda_kv_store_stays_within_its_definition`（bf16、fp16、fp32）：1 个、13 个、2000 个 token，int64 和 int32 的槽号，V 是 QKV 投影的 strided 切片；整个池和参照实现逐位相同。两条回退路径（dtype 不同、一行 12 字节）也是。
  - `test_triton_kv_indices_stays_within_its_definition`：跨多个 512 的块、1 个 token 的行、buffer 的尾部不动。
  - worker 栈的测试里加了断言：`auto` 选的是 `cuda.kv.store@1` 和 `triton.attn.kv_indices@1`；带着请求重载 kernel 单元时把 `kernels.cuda` 也重载两次。
  - CPU（进 CI）：`attn.kv_indices@1` 的参照实现对着手写的列表查；一行单独算和夹在别的行中间算结果相同。

## 没做的

- **GDN 输入投影的双 stream 并行没有做。** SGLang 在录 graph 时把很小的 `in_proj_ba` 放到第二个 stream 上和 `in_proj_qkvz` 并行；这需要 `cuda_graph` 平台提供辅助 stream，按约定本分支不改平台层。SM100 64 层上 1 并发距 SGLang 剩下的 0.32 ms 主要就是这一项加跨进程调度。
- 没有编译器的机器上，`kernels.cuda` 加载失败，引擎起不来。这和 FlashInfer 第一次 plan 时的要求一样，只是提前到了启动时。
- prefill 仍然不走 graph（2000 token 长输入在 64 层 SM100 上已追平 SGLang；32 token 短输入在 1–4 并发下仍有约 25 ms 的 eager host 启动差距）。
