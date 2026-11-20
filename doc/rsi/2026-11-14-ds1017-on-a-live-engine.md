# DS 1017 单 rank：怎么做的，在跑着的引擎上验了什么（2026-11-14）

- 记录人：Claude Fable 5.2（Claude Code）。前半段（kernel、层、模型的第一版）是同一个会话里做的。
- 对应：[../specs/v1-ds1017.md](../specs/v1-ds1017.md) 的阶段 1 和阶段 2 的一部分；PR #65。
- 机器：RTX 4090 D（24 GB）；真实形状和真权重在 SM100 上通过任务服务跑。
- 模型：随机权重的小配置（1024 宽、16 头 × 128、8 层、每种层都有），checkpoint 格式和真的一样。

## 结论

1. **这次不是从头到尾在引擎上热改出来的。** DS 1017 是新的一族模型，它的模型单元写出来之前没有引擎可起。
   - 前面的开发用的是进程内的 worker 栈（`tests/units/support/ds1017.py` 的 `DSWorker`），对着 checkpoint 自带的参照实现一个 kernel 一个 kernel、一层一层地核对。
   - 单元能跑以后才起了整个引擎，在上面做了下面的热改验证。
   - 之后在 DS 上继续开发，就可以按默认方式在引擎上热改了。
2. **热改都提交了，结果和冷启动一致**（见下表）。没有发现更新机制的问题，没有写 framework-bugs 报告。
3. **框架本体没有改。** 改的都是单元和部署工具：
   - `core.plan` 和 `executors.pytorch` 原来写死了 Qwen 的层和状态种类，现在从模型拿（`model.backends`、`model.stores`）。
   - `state.kv_paged` 原来只放 `k`、`v`，现在一层声明什么张量就放什么。
   - 部署工具在拓扑里把 worker 的单元列出来，一个 worker 只装一族模型的单元。
4. **有一条和机制有关的经验**：两个模型单元都提供 `model@1`，同一个 worker 里只能有一个。拓扑的角色不列单元时，会把所有 `process = ["worker"]` 的单元都装进去，第二个模型单元加载时报服务冲突。所以现在起引擎必须在拓扑里列出 worker 的单元（`deploy/rsinfer.py` 已经这样做）。

## 在引擎上做的验证

起法：`deploy/rsinfer.py serve <随机 checkpoint，带真 tokenizer> --max-context 1024 --max-rows 4 --cuda-graph auto`。每次比较用同样三个提示词（英文、单词、中文），贪心，各 24 个 token，比三段文本合起来的摘要。

| 做了什么 | 计划 | 结果 |
|---|---|---|
| 冷启动 | | 起来了。eager 步用 `tilelang` 后端，graph 桶 1、2、4 用参照 kernel（TileLang kernel 不能录进 graph）。`/v1/completions` 和 `/v1/chat/completions` 都能用，对话模板是真 checkpoint 的 |
| `apply --config '{"dsv4.backend": "reference"}'` | 重载层单元和 `core.plan` | 提交。文本变了：参照 kernel 和 TileLang kernel 末位不同，随机权重下几层之后就是另一个 token |
| `apply --config '{"plan.cuda_graph": "off"}'` | 重载 `core.plan` | 提交。文本和上一行**相同**：参照 kernel 下 graph 回放和 eager 逐位一致，经过真调度器和重叠一步也成立 |
| 两个键改回 `auto` | | 提交，文本回到冷启动时的 |
| 改 `kernels/ds1017/ops.py` 的一行（结果不变的写法），`apply --source .` | dry-run：worker 重载 `kernels.ds1017` 和 `core.plan` | 提交，文本不变 |
| 把那一行改回去，再 apply | 同上 | 提交，文本不变 |
| 停掉；冷启动 `--ds-backend reference --cuda-graph off` | 冷启动：对照热改的结果 | 文本和热改成 `reference` 的两次相同 |

## 在 SM100 上的热改会话

SM100 只能通过任务服务用，每个任务都是冷启动。为了在上面也能热改，写了一个会话式任务（`tests/gpu/sm100_session.sh`）：任务起一次引擎，然后跟着一个分支，每推一个提交就 `apply` 一次、跑一次检查。

| 推的提交 | 结果 |
|---|---|
| 什么都没改 | apply 1 秒，提交；记下三个提示词的文本摘要 |
| 改了 `kernels/ds1017/ops.py` 一行（结果不变） | apply 4 秒，提交；文本摘要相同 |
| 标题带 `[session stop]` | 会话结束，任务成功 |

- 第一次提交这个任务时引擎没起来：会话用的随机 checkpoint 没带 tokenizer，前端起不来。脚本当时在退出时把运行目录删了，所以原因是推断的，没有日志为证；改成带上真 tokenizer 以后就起来了。现在起不来时会把引擎和各进程的日志留在结果包里。
- 引擎进程在节点上用的是参照 kernel，不是 `tilelang`：节点代理把每个进程的 `PYTHONPATH` 设成快照目录，装在 Python 环境旁边的库它们看不到。这是框架本来的设计（进程只从快照和环境里 import），没有改。

## 开发中查到的几件事（都和数值有关）

这些不是更新机制的问题，记在这里是因为它们决定了测试怎么写。细节在 [../specs/v1-ds1017.md](../specs/v1-ds1017.md) 第 8 节。

- **随机权重的 DS 模型对末位极其敏感。** 把任何一个 kernel 换成末位不同的另一种实现，8 层之后 logits 能差 0.3 到 0.7（logits 的幅度约 4）。原因是模型在每一步之间都舍入到 bf16 和 FP8，还按分数最高的几个专家路由。所以"在容差内"的端到端比较没有意义，只能逐位比。
- **torch 在 GPU 上的矩阵乘和归约，结果的末位随行数变。** `F.linear` 换一个行数，85% 的 fp32 元素末位不同；`mean(-1)` 在 16 行以下也不同。在 CPU 上，`exp`、`sigmoid` 这类函数的末位也随张量大小变。
- 因此参照 kernel 自己按固定顺序求和、每行单独相乘。这样一个 token 的结果不随这一步怎么排（分块、多行、graph 桶）而变，测试才能逐位比。
- **官方参照实现的 attention kernel 在真实形状（64 头 × 512）下，4090 D 的共享内存不够。** 本地只能用小形状对照，真实形状在 SM100 上跑。
