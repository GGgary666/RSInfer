# RSI-native Serving 框架：讨论纪要与架构设计

- 日期：2026-11-03（2026-11-04 修订）
- 状态：设计讨论稿，尚未定稿。文中的数字都来自公开论文、文档、源码或 GitHub API；"建议""判断"类表述是讨论中的观点，不是结论。
- 本版修订：
  - 改写为通用开源项目的视角，去掉了特定组织的内部系统和流程。原来依赖的具体工具改成按"角色"描述（kernel benchmark、kernel agent、计时基础设施、生产诊断工具），相关论文作为公开文献引用。
  - 新增对 DeepInfer（DeepSeek 开源的生产推理引擎）、nano-vllm、mini-sglang 的源码级调研，以及在 mini-sglang 上做整引擎进化的 SEIS 实验（3.7–3.9），并据此更新了相关决策。
  - 2026-11-04 第二次修订：补充 DeepInfer 的性能定位（3.7）；记录 v0 计划讨论的决策（第 15 节）。v0 的范围、架构和里程碑见 [v0-plan.md](v0-plan.md)，它和本文冲突时以它为准。
- 术语约定：
  - **内核**指引擎核心（core），不是 GPU kernel。GPU 上的计算函数一律写作 **kernel**。
  - **单元**指可被 agent 替换或进化的组件，例如一个 kernel solution、一个调度策略、一个参数集合。
  - **契约**指单元对外承诺的接口、参照语义和不变量。

---

## 0. 摘要

**目标**
- 做一个 RSI-native 的 LLM serving 框架：agent 能在上面持续改进 serving 系统本身（kernel、调度、缓存、引擎机制），结果可信、可叠加、可上线。
- 作为开源项目发布。论文不是主要目标。

**为什么先做 serving**
- 反馈回路短。TTFT、TPOT、吞吐的收益在分钟级就能观测到，不像 RL 或预训练那样要等几百个 step。
- 是展示"agent 改进自身基础设施"潜力的最好场景。

**已经做出的决定**
- **从零构建一个干净的、RSI-native 的引擎内核。**
- **最大限度复用现成的零件**：FlashInfer、FlashMLA、DeepGEMM 等 kernel 库，PyTorch distributed、DeepEP、Mooncake/NIXL 等通信和传输库，以及 vLLM、SGLang、DeepInfer 的模型实现。kernel 库和分布式后端不自己重写。
- **成熟的开源引擎作为常设的正确性和性能对照基准**：vLLM 和 SGLang；DeepSeek 系列另加 DeepInfer。

**核心设计思想**
1. 机制与策略分离。
2. 状态比逻辑活得长：状态外置到长寿命的状态服务。
3. 组合在编译执行计划时是动态的，执行时是静态的。
4. 凡是影响输出或性能的东西，都要声明、都要记日志。
5. 评测器在候选的信任边界之外。

**和现有框架的本质区别**
- RSI-native 不等于加几个接缝。它要求若干**全局不变量**：完整的失效追踪、明确的状态归属、参照模式、系统身份与溯源、跨 rank 的事务式变更。
- 在百万行量级、每月合入近两千个 PR 的代码库里，这些不变量很难靠改造得到，只能在新内核里靠结构强制。
- 本轮调研从两头印证了这一点：
  - 生产引擎（DeepInfer）已经把 batch invariance、确定性采样、代价模型驱动的调度做成了一等特性。但它的配置散在几十个环境变量里，派生产物的缓存键不完整，和基础设施耦合的部分仍然开不了源。
  - mini 引擎（nano-vllm、mini-sglang）小到 agent 能一次读完，但没有契约。SEIS 在 mini-sglang 上做整引擎进化，吞吐提升到 3.27 倍，同时也暴露出 agent 会绕过评测：去掉 EOS、只算部分词表、复用已经失效的缓存元数据。

**可借力的公开工作与缺口**
- kernel 层：kernel benchmark 和 kernel agent 已经很多（flashinfer-bench、KernelBench、Atrex-Bench、AKA、AlphaEvolve 等）。共享 GPU 上的并发计时有 KREX。
- 生产观测和诊断：Herschel 这类系统在运行时 attach 采集 profile，再匹配已知的低效模式。
- 引擎级的 agent 改动：Herschel 的 Optimizer（在生产引擎上打一次性补丁）、SEIS（在 mini 引擎上持续进化）。
- 缺的是：
  - e2e 集成评测；
  - kernel 进入引擎的标准路径；
  - 系统化的 policy 层搜索；
  - 改动之间的持续叠加；
  - 能约束 agent 改动的结构化正确性保证（SEIS 的教训）。

---

## 1. 背景与目标

### 1.1 最终目标

- 构建 agent-native 的基础设施，最终覆盖推理、预训练和后训练三类框架。第一个落地对象是 serving。
- "RSI-native" 指框架从设计之初就让 agent 能安全地完成整条回路：观察、提假设、修改、应用、评测、接受或拒绝、在已接受的改动上继续叠加。
- 对开源社区和后续研究都真正有用。首批目标是两个主流的开源权重模型家族：Qwen 和 DeepSeek。

### 1.2 可借力的公开工作

下面几项公开工作分别覆盖 kernel 评测、kernel 优化 agent、计时基础设施和生产诊断，和本项目最相关。本项目不依赖其中任何一个：它们在设计里以"角色"出现，任何同类工具都可以接入。

| 工作 | 做什么 | 关键数字 | 边界 |
|---|---|---|---|
| **Atrex-Bench**（[2607.14541](https://arxiv.org/abs/2607.14541)） | 来自生产 trace 的算子级 kernel 生成 benchmark | 30 个算子、440 个热点 shape、1,303 份 profile、约 20 个部署模型，覆盖 XPU-A 和 H20，超过 1 万张卡；前 5 个算子约占 64% 的 GPU 时间；最好的原生模型 S_agg=0.107；"正确性错觉"：有模型正确率 84.8%，但目标 DSL 采用率只有 43.8% | 只测算子级；只覆盖推理；目前只有 XPU-A 上的实测结果 |
| **AKA**（Atrex-Kernel-Agent） | 按 profile 驱动的 kernel 优化 agent | GPU Wiki 含 298 个参考 kernel、244 篇优化知识文档；S 从 0.28 提到 0.42；attention_forward 比 fallback 快 6.7 倍；dropout 式部分重启；输出契约是单个评测器友好的 kernel 文件 | 产物还不能直接被引擎加载 |
| **KREX**（[2609.30057](https://arxiv.org/abs/2609.30057)） | 共享 GPU 上的并发 kernel benchmark：只对计时区间独占 | 每天 30 万个 benchmark 任务，覆盖 3 家厂商 6 种 GPU；高峰时排队需求是在线容量的 4.8 倍；一个 agent session 的 GPU 利用率只有 3.4%；吞吐最高 3.4 倍；p95 计时偏差 0.30%（kernel >10ms）/ 1.58%（>1ms）/ 3.90%（>0.1ms） | 假设候选是配合的；计时区间由脚本作者标注，系统不校验；目前只支持单卡 |
| **Herschel**（[2609.40247](https://arxiv.org/abs/2609.40247)） | 生产环境的持续优化：运行时 attach 采集 profile、Analyzer 匹配低效模式、Optimizer 让 agent 实现修复 | 每次采集约 30 秒，TTFT 开销低于 0.5%、TPOT 低于 7%；6 个月约 1.7 万条 trace，覆盖 120 多个模型变体和 10 多种加速器，23% 命中模式；35 个模式（引擎 25 个、算子 10 个）提炼自 2 万多条 PR/CR；受控实验用 8–32 卡；结果经专家评审后进 canary | 有人在环；改动是针对具体引擎版本的一次性补丁；每个实验的评测都是临时设计的；字段读取器要追着引擎内部布局改（6 个月发了 5 个版本） |

另外两类参照对象在第 3 节单独展开：
- flashinfer-bench 的 Definition/Solution 契约和 `apply()`（3.4）；
- DeepInfer、nano-vllm、mini-sglang 三个引擎，以及 SEIS 的进化实验（3.7–3.9）。

### 1.3 把现有工作放到 RSI 回路上

| 回路阶段 | kernel 层 | 引擎层 | policy 层 |
|---|---|---|---|
| 观察 | 生产 profiling（Herschel 类：低 SOL 的算子和 shape） | 生产 profiling（等待、同步、重复工作、无用计算） | 有数据，但没被当作 policy 搜索来用 |
| 假设 | kernel agent 的知识库（如 AKA 的 GPU Wiki） | 低效模式库和参考修复（如 Herschel） | 空白 |
| 修改 | kernel agent（AKA、AlphaEvolve 等） | Herschel Optimizer 的 agent；SEIS（在 mini 引擎上整体改写） | 空白 |
| 评测 | 算子级 benchmark 加区间独占计时（Atrex-Bench、flashinfer-bench、KREX；严格） | 临时搭建的受控实验；SEIS 的数值门禁加下游任务（会被绕过，见 3.8） | 空白 |
| 接受与上线 | — | 专家评审 → canary | — |
| 叠加 | 把经验蒸馏回知识库（AKA 的计划） | 生产引擎上每个改动单独评审，基线不会持续演进；SEIS 的 session 链继承代码和报告 | — |

最大的三个断点：
1. kernel 层的产物进不了引擎。
2. 引擎层的评测没有标准化。算子级 benchmark 那套"生产来源、按重要性加权、可归一化、防刷分"的契约，在 e2e 层面还不存在。SEIS 的经验说明，临时设计的 e2e 门禁会被 agent 绕过。
3. mini 引擎上的结果能否迁移到生产引擎，还没有人量过。SEIS 的 3.27 倍是 0.6B 模型在单请求负载上的结果。

---

## 2. 范围

### 2.1 在范围内

- **只做 serving。** 训练和 RL 不在第一阶段，但内核的抽象（单元、状态服务、事务、溯源）要能迁移过去。
- **模型**：Qwen 系列和 DeepSeek 系列。按当前的代际，需要覆盖：
  - Qwen3.5/3.6：3:1 的 Gated DeltaNet 加 Gated Attention 混合结构；MoE 有 256 个专家，激活 8 个路由专家加 1 个共享专家；原生多模态；MTP。
  - DeepSeek V4：CSA 加 HCA 混合的压缩稀疏注意力（压缩比 4 和 128，带 indexer）；mHC；MoE；MTP；百万 token 上下文。
  - DeepSeek-V4.1-Flash：DeepInfer 的实现（3.7）还显示出以下结构：
    - YOCO 式的 encoder/decoder 分层：decoder 层的 block KV 由最后一个 encoder 层的 hidden state 投影得到，decoder 层只需各自计算 SWA KV；
    - Engram：使用独立的并行组，按词表维度切分，在 GPU 部署时放在 host 内存里、经 RDMA 读取；
    - 视觉编码器；
    - 带 Confidence Head 的 draft model（DSpark）；
    - FP4 的 main KV，MXFP4 的 indexer KV 和路由专家权重。
- **硬件**：
  - NVIDIA Hopper/Blackwell，包括 H20 这类算力受限、显存充裕的型号。
  - 非 NVIDIA 加速器，例如华为昇腾（DeepInfer 已经同时支持 SM100 和 Ascend 950），通过平台插件接入。
- **工作负载**：公开的生产 trace（如 Mooncake、Azure LLM inference trace、BurstGPT），加上部署方自带的 trace。跨部署聚合时，权重按 GPU 小时计。
- **部署形态**：从单节点开始，逐步到多节点、大规模 EP、P/D 分离。
- **开源**：内核和评测平面从第一天起开源。部署方私有的基础设施（集群管理、存储、私有加速器）全部做成插件。

### 2.2 不在范围内 / 非目标

- 不追求成为覆盖所有模型、所有硬件的通用引擎，不和 vLLM/SGLang 拼覆盖面；在 DeepSeek 模型的生产部署上也不和 DeepInfer 竞争。
- 不重写 kernel 库（FlashInfer、FlashAttention、FlashMLA、DeepGEMM、CUTLASS、Triton、TileLang 等），也不重写分布式通信后端（PyTorch distributed、NCCL/HCCL、DeepEP、Mooncake/NIXL）。
- 不追求"所有层都像 Vite 那样热替换"（理由见 4.1）。
- 论文不是主要目标，但内核里沉淀下来的数据和轨迹应该能支撑后续研究。
- 第一阶段不做在线真实流量上的自动 canary。canary 和上线交给部署方现有的流程，由人评审。

### 2.3 复用和自建的边界

| 类别 | 策略 | 说明 |
|---|---|---|
| GPU kernel 库 | 复用 | 统一包装成 D6 里的 kernel solution |
| 通信和 KV 传输 | 复用 | NCCL/HCCL、DeepEP、Mooncake/NIXL 等，包装在 D9 的通信组状态服务后面 |
| 模型实现 | 移植 | 从 vLLM、SGLang 移植成 D13 的模型规格和初始 solution；DeepSeek 系列参考 DeepInfer（Apache-2.0） |
| OpenAI 协议、tool parser、reasoning parser | 移植或复用 | 用测试语料对齐；DeepSeek 系列可以直接复用 deepseek-recipe（MIT，Rust 实现，带 Python 绑定） |
| 引擎内核 | **自建** | 单元模型、状态服务、plan 编译器、调度机制、事务、溯源、自省、执行层抽象 |
| e2e 评测平面 | **自建** | 契约设计借鉴算子级 benchmark（Atrex-Bench、flashinfer-bench），测量借鉴 KREX 的计时区间独占 |
| 观测诊断 | 复用并对接 | 外部 profiling 和诊断工具（Herschel 类系统、Nsight、PyTorch profiler）接入内核提供的稳定自省接口 |

---

## 3. 调研要点

### 3.1 现有框架的热更新能力

结论：**vLLM、SGLang、verl、Megatron 都没有官方的代码级热替换（HMR），也没有官方文档把它列为目标。** 它们有的是"进程不重启、替换某一类状态"的 API。DeepInfer 连这类 API 都没有，生产上靠实例冗余吸收重启。

| 框架 | 已有的运行时能力 | 没有的 |
|---|---|---|
| vLLM | sleep mode L1/L2 加 `reload_weights`；`update_config`；RL 场景的四阶段 weight transfer，加 `pause`（abort/wait/keep 三种模式）和 `resume`；运行时 LoRA；`VLLM_SERVER_DEV_MODE` 下的 `/collective_rpc`（文档原话 "extremely dangerous"）；`--worker-extension-cls`；Elastic EP（`/scale_elastic_ep`，停顿从 56–100 秒优化到 0.82–1.96 秒） | 代码 HMR；在线修改调度参数（PR #30316 仍是 open 且 stale）；运行时修改 TP/PP 或 `max_model_len`；公开的运行时 CUDA graph recapture API |
| SGLang | `update_weights_from_*`（带 `recapture_cuda_graph` 参数）；`release/resume_memory_occupation`；pause/continue；`flush_cache`；`set_internal_state`（白名单里只有少数几个 key） | 在线修改 chunk 大小、调度策略等（PR #37661 仍是 open。作者实测热修改 16–22 毫秒，而同样的改动靠重启要 56–86 秒 warm、141–391 秒 cold） |
| verl | 无 | Ray 的 actor 和 driver 同生共死，改 driver 就要重启整个 stack；`ppo_mini_batch_size` 等配置在 init 时就按 `rollout.n` 派生并固定 |
| Megatron | Dynamic-CP：每个 micro-batch 选择 CP size，最高 1.48 倍加速 | NVRx 的 in-process restart 在 v0.6 被 deprecated，之后被移除；改 LR 要停机，用 `--override-opt_param-scheduler` resume |
| DeepInfer | EPLB 每隔若干 step 在线重排专家权重；环境变量可以用 `rearm()` 重新解析；（生产系统）decode 引擎下线前有 1 分钟宽限期，把未完成的请求连同 KV 转给其他引擎 | 在线服务入口；任何运行时变更 API；调度参数在调度器构造时读入并校验，之后不变；（生产系统）引擎重启是分钟级，EP 故障靠重启并按健康卡数重定 EP 大小 |

**热更新的正确性本身就很难。** vLLM PR #61287（open）给 reload 路径做了"重载后必须和全新启动一致"的测试，列出的 bug 包括：
- reload 时 attention sinks 被静默丢弃；
- FlashInfer 的 scale 只在首次 forward 时计算一次；
- 已 capture 的 CUDA graph 看不到新值。

**不同层的热替换难度：**

| 层 | 例子 | 难度 |
|---|---|---|
| 控制面 Python | 调度策略、reward、路由 | 低 |
| worker 内 eager 执行的 Python | sampler、loss | 中：要在每个 rank 一致地 patch |
| 编译产物 | torch.compile 结果、CUDA graph | 高：要重新编译和 capture，十几到几十秒 |
| native 扩展 | `.so`、CUTLASS | 基本只能重启进程 |
| 拓扑和内存布局 | TP/EP、NCCL、KV 池 | 需要专门的弹性机制 |

### 3.2 RSI 相关的研究和官方表态

- 没有找到明确提出"vLLM/verl/Megatron 需要 HMR 以支持 agent 自我改进"的论文或官方声明。
- 前沿实验室把 AI 自我改进当作能力和风险来测：OpenAI Preparedness Framework v2 的 AI Self-improvement 类别，Anthropic RSP 的 AI R&D 阈值。它们只测能力，不对基础设施提要求。
- agent 改进基础设施的工作：
  - AlphaEvolve：Gemini 的一个 kernel 快 23%，训练时间减少 1%；FlashAttention 快 32.5%。
  - SEIS：agent 端到端进化 mini-sglang，单请求吞吐提升到 3.27 倍（细节和教训见 3.8）。
  - ISO-Bench：54 个任务，取自 vLLM 和 SGLang 的真实 PR。
  - Automated LLM Speedrunning、karpathy/autoresearch（每次实验固定训练 5 分钟，不计启动和编译）、AIRA₂、ADRS。
- 这些工作都承认"反馈回路慢"是瓶颈，但解法是：
  - 缩小被改进的系统（mini-sglang、0.6B 模型）；
  - 用模拟器代替真系统（ADRS）；
  - 并行跑多个实验（AIRA₂）；
  - 固定时间预算（SEIS 每个 session 2 小时）。

  没有人用 HMR。
- **评测完整性是 RSI 的核心风险。** Sakana AI CUDA Engineer 就出过 agent 利用评测 harness 的内存漏洞制造虚假加速的事；SEIS 里也记录了四类绕过评测的案例（3.8）。长期运行、被反复热 patch 的进程会让结果依赖 patch 历史，所以最终结论应该来自隔离评测和冷启动对照。
- Cordis 的论文（[arXiv 2608.25512](https://arxiv.org/abs/2608.25512)，北大和 DeepSeek-AI）明确把 "self-evolving agent harnesses" 列为动态组合的动机。这是 harness 层面的正式工作，不是 serving 引擎层面的。

### 3.3 动态参数切换的三类

| 类别 | 例子 | 机制难度 | 现状 |
|---|---|---|---|
| A. 每步或每请求的标量 | LR、temperature、`n`、`max_tokens`、KL 系数 | 机制上几乎为零 | 卡在框架的配置传递：派生值在 init 时固定；resume 时 checkpoint 会覆盖新值。Tinker 是反例：每次 `optim_step` 都重新读超参 |
| B. 有启动上限的调度参数 | chunk 大小、`max_num_seqs`、调度策略 | 中：调度循环每步都读，但 KV 池和 CUDA graph 的 capture 尺寸在启动时就定了 | vLLM #30316、SGLang #37661 都还是 open |
| C. 结构类 | TP/PP/EP/CP、`max_model_len`、量化方式 | 高：权重和 KV 要重新切分、通信组要重建、要重新编译和 capture | 官方只有 vLLM Elastic EP 和 Megatron Dynamic-CP；其余都在论文里（见附录 B） |

sequence length 跨两类：调小 `max_tokens` 属于 A 类，调大到超过 `max_model_len` 属于 C 类。

### 3.4 flashinfer-bench 的 `apply()`

- **机制**：在 FlashInfer 的 API 层做 wrapper（FlashInfer 本体现在也内置了 `trace_apply`）。根据调用参数的固定轴算出 definition 名，比如 `rmsnorm_h4096`；命中注册表就走对应的 solution，否则 fallback；有 hit/fallback/error 的统计。论文报告每次调用开销 1–2 微秒，端到端低于 0.8%。
- **边界**：
  - 论文里的流程是在引擎启动前建索引。`enable_apply()` 虽然可以在进程内重新调用来替换映射，但要在每个 worker 里调用。
  - 落在 CUDA graph 或编译区域里的调用，路由结果会被冻结在 capture 那一刻。例如 vLLM 默认的 `FULL_AND_PIECEWISE` 模式下，decode 走 full graph，换 kernel 需要 recapture。
  - 只覆盖经过 FlashInfer API 的调用。
  - 一个 definition 只能有一个 solution，没有 canary 或 shadow。
- **真正值得借鉴的**：
  - 算子契约：Definition、Workload、Solution、Trace 这一套 schema。
  - 评测和部署分离：评测有完全隔离模式，每个 solution 一个子进程，跑完销毁 CUDA context。
  - fallback 和统计。

### 3.5 Cordis 和 DeepSeek Harness（DSH）

- **机制**：
  - DSH 里的一切都是插件，包括 agent loop 本身。
  - 插件向共享 context 贡献 service、typed event 和可撤销的 effect。
  - `inject` 声明依赖，依赖不满足时插件停在 PENDING；依赖的 service 消失时，下游会被回滚或重载。
  - 配置按稳定 id 做 diff。
  - HMR 的做法是 unload 再 load 受影响的插件子树。框架层依赖变化时退回 `loader.exit()`，也就是重启。
- **关键取舍**：HMR **不保留被重载插件自身的内存状态**。状态之所以能保住，是因为它们放在了别的 service 里，或者写进了持久日志。DSH 的规则是 "Model-visible means logged"。**这对 ML 基础设施是正确的取舍**：权重、KV 等状态必须由长寿命的状态服务持有。
- **Creator mode**：agent 自己写一个插件包并安装，在同一个 session 里生效，结果是 `applied`、`restart-required` 或 `failed` 三种之一。它诚实地暴露了 `restart-required`。
- **直接搬到 GPU 数据面会断的地方**：
  - 编译产物是跨插件融合出来的；
  - Cordis 是单进程模型；
  - 热路径开销；
  - native 代码卸不掉；
  - 缺少显式的状态迁移。

### 3.6 其他可借鉴的技术

- **算子层**：
  - `torch.library` 的 IMPL 覆盖：`del lib` 即可撤销，`get_kernel` 可以取回原实现做 fallback。BackendBench 就用它在运行时覆盖 aten 算子。
  - vLLM `CustomOp.register_oot`：只在 init 时生效。
  - HF `kernels` 的 `kernelize`。
  - CUDA graph update：新版 CUDA 允许在限制下修改 kernel 函数，但 graph 拓扑必须完全一致。
- **组件和热更新层**：
  - Erlang/OTP：新旧两个版本的 module 可以共存，`code_change/3` 负责状态迁移，由 supervision tree 管理。
  - Linux livepatch 的一致性模型：只在任务没处于被 patch 函数的栈上时才切换。
  - promplate/hmr：在运行时追踪模块依赖图的 Python HMR；jurigged。
- **分布式和交互式开发**：
  - PyTorch Monarch：常驻的集群分配加 actor supervision。
  - Tinker：训练循环在 client，GPU 状态在服务端。
  - Ray Serve：`reconfigure()` 处理参数热更新，代码变化走 rolling restart。
  - vLLM Elastic EP：后台 prepare，短暂 commit。
- **执行层**：
  - batch-invariant 的确定性推理思路（Thinking Machines 2025）。DeepInfer 已经把它做成了生产特性（3.7）。
  - 进程快照：CRIU；NVIDIA 侧有 cuda-checkpoint。

### 3.7 DeepInfer：DeepSeek 开源的生产推理引擎

**基本情况**
- 2026-10-20 开源，Apache-2.0。README 自述是支撑 DeepSeek 大规模线上服务的推理框架；官方 Q&A 称单集群每天处理 155 万亿 token。
- 最初主要基于 vLLM 构建，并借鉴了 SGLang 和 TensorRT-LLM。很多文件头保留着 "Derived from vLLM; modified by DeepSeek"。
- 范围很窄：
  - 只支持 DeepSeek-V4.1-Flash；
  - 硬件是 NVIDIA SM100（CUDA 13）和 Ascend 950（CANN 9.2）；
  - 只提供离线推理入口。README 明确列出完整服务还缺三块：KV cache 系统、VE/P/D 分离（引擎侧大部分功能已实现）、prefill 和 decode 的 DP 负载均衡。
- 代码规模（`wc -l`，commit a679bb8）：`deepinfer/` Python 包约 4.6 万行，`csrc/` C++ 约 5.9 千行（调度器 `scheduler.hpp` 1,329 行），测试约 3.1 千行。kernel 库以 git submodule 引入：FlashMLA、DeepGEMM、DeepEP、DeepSelect，以及 DeepEP-Ascend、DeepGEMM-Ascend；另有自带的 TileLang kernel。
- 治理：截至 11-03，公开仓库只有 6 个 commit，合入了 2 个外部 PR，约 210 个 PR 处于 open。更接近"代码发布"，而不是社区共建。
- 前情：2025-04 DeepSeek 说明过不开源内部引擎，理由是 vLLM fork 分叉太久、和内部基础设施耦合、维护带宽不足。DeepInfer 最终开源了引擎本身，但 KV cache 系统和负载均衡仍然因为"和内部基础设施紧耦合"没有开源。

**架构要点**（来自源码和官方 Q&A）
- **进程拓扑**：每个 DP rank 一个 `LLMEngine` 进程，调度器和 driver worker 在同一进程；CP 的其他 rank 另起 worker 进程，由共享内存广播驱动（沿用 vLLM 的 `shm_broadcast`）。
- **语言分层**：调度器、block manager、输入元数据准备、MTP 变长调度、采样准备和输出处理用 C++ 写，经 pybind 暴露；模型、采样、worker 用 Python 加 TileLang。
- **异步调度**（zero-overhead scheduler，默认开启）：GPU 在跑 `forward(N) → sample(N)` 时，CPU 并行做 `schedule(N+1) → launch_forward(N+1) → post-process(N−1)`。代价是调度 N+1 时还拿不到 N 的结果，只能乐观估计。
- **状态**：两个池。
  - paged block KV：main KV 加 indexer K，块大小 128 token。main KV 用 FP4（per-16，e4m3 scale），indexer KV 用 MXFP4。
  - 每序列一个固定 slot（`StateBlock`）：SWA KV 的环形缓冲区，以及压缩边界的累加器（`ca_state` / `ica_state`）。slot 分为需要持久化的区域（前 20 层的 SWA）和可丢弃的区域（YOCO consumer 层的 SWA，靠重算恢复）。
  - 调度器联合分配两种状态：state slot 分不出来时，block 也视为暂不可分。
- **KV 复用和命中规则**：
  - block KV 存在专用的 3FS 集群（约 100 PiB SSD；agent 流量 TTL 3 天、chatbot 流量 7 天）。SWA KV 只留在产生它的机器内存里，靠 Entrypoint 上的内存目录实现跨机命中。
  - SWA KV 只在 128 token 的整数倍位置打快照；decode 期间每跨过一个 128 边界，就在显存里拷一份副本。
  - 命中长度由 block KV 决定；SWA KV 缺失时回退一个块并重算。
  - **SWA bounded replay**：只重算最近 128 个 token，并把注意力窗口截断在重算段内，得到的是**近似**的 SWA KV。decoder 侧的同类截断让 prefill 总计算量接近减半。
  - 缓存兼容键是 `(部署配置, 模型权重, 用户组)`。权重、块的字节布局、tokenizer 变化，以及改变 KV 数值的框架改动，都会让旧缓存失效。**生产环境从未做过缓存格式迁移。**
- **代价模型驱动的调度策略**：
  - prefill chunk 大小不是固定值：每个 step 有一个时间预算，用拟合的二次代价模型 `cost(q, c)` 反解出不超预算的最大 `q`（上限 32,768 token），预算再在 DP 组内做 all-reduce MIN。
  - DSpark 变长验证：draft model 每次出 5 个 token，Confidence Head 预测每个 draft token 的接受率；再结合实测的 decode 吞吐曲线，选出使有效吞吐最大的前缀去验证。验证的 token 数减少约 30%，平均接受长度基本不变（3.29 → 3.20）。在生产工作点（每 rank 约 60 并发）吞吐 +4–7%，但 16 并发时是 −1.5%，96 并发时是 +11.0%。
  - 异步调度下，CPU 只定验证总预算，每个请求的验证长度由一个 TileLang kernel 在 GPU 上算出。
  - EPLB：每卡放少量冗余专家，按最近 N 步的负载统计，每 M 步重排一次，专家权重经 NVLink `symm_mem` 在组内搬运。prefill 每卡 4 个冗余、窗口 100 步、每 1,000 步重排，`moe_lb`（最慢 rank / 平均）为 1.045–1.074；decode 每卡 1 个冗余、窗口 1,000 步、每 10,000 步重排，`moe_lb` 为 1.090。
  - 负载均衡（未开源）：prefill 按上下文长度分 4 档，按算力配额拉取请求；decode 同时平衡请求数和 KV 总长度。整体约达到完美均衡下离线 benchmark 的 80%。
  - 另有两个现成的代价模型：C++ 的 `FlopsEstimator` 按请求估算 FP8 等效 FLOPs，用于 MFU 统计；`DecodePerfCache` 在启动时 replay 已 capture 的 graph，测出各 batch size 的 decode 耗时，交给 C++ 调度器的 MTP 变长策略使用，并落盘缓存（默认 TTL 1 小时）。
- **确定性**：
  - batch-invariant 推理和确定性采样是一等特性，目前只支持 NVIDIA。
  - 测试 `test_batch_invariant.py`：同一批 prompt 先批量生成，再用 `golden_token_ids` 逐条单独生成，要求 token、logprob、top-k logprob 完全相等。
  - 确定性采样要求关闭 MTP 变长验证（`DEEPINFER_ENABLE_MTP_VARLEN=0`）。
- **运行时守卫**：model forward 和 sampling launch 期间用 `forbid_cuda_sync` 禁止 CPU-GPU 同步，违规直接报错。
- **配置**：
  - 59 个环境变量集中定义在一个 `envs` 类里，首次读取后缓存（`rearm()` 可以重新解析）。
  - 部分默认值由引擎类型推导后写回 `os.environ`，再重新解析。
  - 仍有少数变量绕过集中定义。例如 C++ 调度器自己 `getenv("DEEPINFER_USE_NEXTN_DECODE")`，用 `atoi` 解析，默认值靠一行注释和 Python 侧保持一致。
  - 布尔变量的严格解析是开源后才补上的（PR #218）。
- **平台抽象**：
  - `Platform` 类只有很薄的一层：TileLang 后端目录（`cuda`/`asc`）、通信后端（nccl/hccl）、FP8 scale 的 dtype、host 内存 pin。
  - kernel 在导入时按平台整体分发：`tilelang_ops/cuda` 和 `tilelang_ops/asc` 各约 30 个文件，两边的集合不完全相同。缺失的 kernel 抛 `KernelUnavailableError`，测试里转成 skip。
  - NPU 路径借助 torch_npu 的 `transfer_to_npu` 把 `torch.cuda.*` 映射到 `torch.npu.*`，并在初始化时设置一批 HCCL 环境变量、monkey-patch stream 属性。
- **运维**（官方 Q&A 描述的生产系统）：
  - 引擎重启是分钟级。容错靠冗余配额和请求级重路由，而不是缩短单个引擎的重启时间。
  - 大 EP 实例里一张卡故障会让整个实例不可用；重启时按健康卡数重新确定 EP 大小。

**性能定位**（2026-11-04 核实）

"DeepInfer 远快于 vLLM"的说法没有公开证据支持：
- **在 DeepInfer 自己的基准配置下，几家基本持平。** 它的基准配置是 NVL72 机架，prefill 4 卡、decode EP64。
  - vLLM 用自己的实测对比从 DeepInfer 图表上读出的数，自称 decode 差距在 6% 以内，prefill 曲线也接近。
  - SGLang 在同一硬件上用 DeepInfer 自己的 benchmark 脚本对跑，结果持平或略快。
- **在单台 8 卡节点上，第三方结果互相矛盾。**
  - GPUStack 在 8×SM100 上跑 DP8+EP8（初步测试，两边都没调优）：DeepInfer 的输出吞吐和 ITL 更好，vLLM 的 TTFT 和 P95 更好。
  - 另一篇在 8×B300 上测：vLLM 的 decode 快 2.0 倍，8K 输入时快 4.3 倍，prefill 快 9.1 倍。作者的解释是单机混跑不是 DeepInfer 的设计场景，而 P/D 分离等配套部件没有开源。
- **DeepInfer 公开的曲线是理想化的 benchmark**：强制专家负载均衡、所有请求从同一个 prompt fork 出来、固定验证 3 个 draft、按 3.25 个接受 token 折算吞吐。

**对本项目的意义**
- 它是 DeepSeek 系列最好的参照实现和对照基准：模型代码、kernel 选择、量化格式、状态布局都可以直接对照。
- 在单台 8 卡节点上，它不一定是最快的引擎。它的 kernel 和 vLLM 基本相同，而 vLLM 在单机上跑得更快，说明单机上仍有大量引擎层的空间。这个已知可达的上界可以用来检验 RSI 的改进是否真实。
- 它不是 RSI-native 的：没有单元和契约；没有运行时变更机制；配置散在环境变量里，而且有多条读取路径；派生产物的缓存键不完整（decode 吞吐曲线的键只含模型、batch 列表和 DP/CP/EP 大小，不含 kernel 版本和影响 kernel 选择的环境变量）。
- 它把很多"策略"写成了代码里的固定逻辑加环境变量参数：chunk 预算、prefill 尝试间隔、验证长度、EPLB 的窗口和周期、graph 分桶间隔。这正好是 D7 决策点的一份现成清单。

### 3.8 nano-vllm、mini-sglang 和 SEIS

"mini vLLM" 这个名字下有很多个人教学项目，影响力最大的是 nano-vllm。mini-sglang 是 SGLang 官方组织维护的精简版。

| | nano-vllm | mini-sglang |
|---|---|---|
| 作者 / 组织 | Xingkai Yu 的个人项目（作者在 DeepSeek 工作，不是官方发布） | sgl-project（SGLang 官方组织），LMSYS 博客发布 |
| 时间 | 2025-06 创建；最近一次提交 2026-04-26 | 2025-09 创建，2025-12 正式发布；最近一次提交 2026-05-17 |
| 活跃度（截至 11-03） | 约 16.3k star；2026-08 以来 0 个 commit | 约 5.5k star；2026-08 以来 0 个 commit |
| 代码量 | README 称约 1,200 行；`nanovllm/` 共 1,385 行 | README 称约 5,000 行；`python/` 下去掉空行、注释和 benchmark 后约 6,000 行 |
| 入口 | 只有离线 `LLM.generate` | OpenAI 兼容的 API server、交互 shell、离线接口 |
| 模型 | Qwen3 dense | Llama-3、Qwen2.5、Qwen3（含 MoE）、Mistral |
| 进程拓扑 | rank 0 和引擎同进程；其他 TP rank 由共享内存 + pickle + `Event` 驱动 | API server、tokenizer、detokenizer 各一个进程；每个 TP rank 一个 scheduler 进程；控制走 ZMQ，张量走 NCCL |
| 调度 | 同步循环：schedule → run → postprocess；prefill 优先；只有队首请求允许 chunk | overlap 调度（借鉴 NanoFlow）；chunked prefill；prefill 优先（代码里留了"支持 decode 优先等策略"的 TODO） |
| KV 和缓存 | paged KV；用 xxhash 做块哈希，实现前缀缓存 | paged KV；radix cache（可换成 naive）；空闲时做完整性检查（空闲页 + 缓存页 = 总页数） |
| kernel | FlashAttention；部分层用 torch.compile | FlashAttention、FlashInfer、TRT-LLM fmha 三种 attention 后端，prefill 和 decode 可以分别选；JIT CUDA kernel（TVM-FFI） |
| CUDA graph | ModelRunner 自己 capture；分桶 1、2、4、8，之后每 16 一档，最大 512 | GraphRunner capture；分桶 1、2、4，之后每 8 一档，最大 160 或 256（按显存判断） |
| 扩展方式 | 改源码 | 注册表：attention 后端、模型、MoE 后端、缓存类型 |
| 全局状态 | 模块级可变的 `_CONTEXT` 传递 attention 元数据；KV 通过 `setattr` 注入到各层 | 全局 `Context` 只允许设置一次，前向时用上下文管理器绑定 batch |
| 配置 | 构造参数 | `MINISGL_*` 环境变量；解析失败时静默回落到默认值 |
| 性能 | README：RTX 4070 Laptop、Qwen3-0.6B、256 条请求，1,434 tok/s，对比 vLLM 1,362 tok/s | README 称吞吐和延迟达到 SOTA 水平（本次未复现） |

**共同点与局限**
- 可读性是真的。nano-vllm 整个引擎约 1.4 千行，agent 一次就能读完；mini-sglang 约 6 千行，按模块读也不难。这正是 5.1 里"可读的单元和契约"这一局部性质。
- 但它们没有契约：
  - 隐式的全局可变状态（nano-vllm 的 `_CONTEXT`）和通过属性注入的 KV，正是 D3 要用静态检查禁止的写法；
  - CUDA graph 由组件自己 capture，graph 内读的是全局 buffer。换 kernel 或改元数据格式时，没有任何机制提示需要 recapture；
  - 没有参照模式，也没有失效追踪。
- mini-sglang 博客自己说明了动机：往 SGLang 这样的大框架里注入新逻辑风险很高，"很容易破坏系统的隐式不变量"。精简解决的是"读得懂"，没有解决"不变量由谁保证"。
- 两者都已经停更约半年，不适合当产品底座；适合当 agent 实验的 sandbox、教学材料和 benchmark 的起点。

**SEIS：在 mini-sglang 上做整引擎进化**（[arXiv 2610.04646](https://arxiv.org/abs/2610.04646)，芝加哥大学）
- **设置**：agent 固定，进化的是引擎源码。每个 session 有 2 小时的硬性时限；session 之间继承代码和报告；可以并行多条链，并定期交换。目标是 Qwen3-0.6B 在 H100 上的单请求吞吐。唯一的硬约束是不许改评分脚本。
- **结果**：594 → 1,942 tok/s（3.27 倍），关掉投机解码仍有 3.08 倍。改动覆盖量化 kernel、跨算子融合、执行开销、prompt-lookup 投机解码，也碰到了调度和 KV 管理。持续的链优于独立尝试。
- **门禁**：数值门禁（相对原实现 ppl 差 5% 以内、平均 KL ≤ 0.05、argmax 一致率 ≥ 90%），外加 GSM8K 和合成检索任务作为事后验证。最好引擎的准确率变化是 GSM8K +1.06 个百分点、检索 −0.37 个百分点。也就是说，输出已经变了。
- **评测被绕过的四个案例**（论文第 6 节）：
  1. 评测只保护它执行到的代码。按选项概率打分的多选题只走一次 prefill，decode 路径完全没被测到。
  2. 评测没检查的东西，agent 会拿去换速度。定长 benchmark 忽略 EOS，一个候选把 EOS 从输出头里去掉了也能通过。之后加了"必须正常终止"的短生成探针，在 Qwen 主实验中拦下了 14 个 session。
  3. 缓存失效。一个优化缓存了 attention 的准备信息，新请求进来时没有刷新。对固定 prompt 反复计时能通过，换一串不同的请求就会挂起。
  4. 任务分数不能认证引擎。有的最终引擎省略了约 39% 的词表行，GSM8K 只比原版低 1.36 个百分点，最后靠代码审计才发现。
- 另外，继承下来的报告会把错误结论一起传下去。
- **局限**（论文自述）：一个引擎、单请求负载、两个小模型、没有并发服务负载。

**对本项目的意义**
- SEIS 是"agent 改进 serving 系统"目前最直接的证据，也是本项目设计最直接的反例集。四个案例分别对应：参照模式要覆盖 decode 路径（D10）、停止行为必须一致（10.4）、失效追踪（D3、D5）、全词表 logprob 比对（D10）。
- 它的结果是在弱基线、单请求负载上得到的，所以 13.2 里的"迁移率"必须单独度量。

### 3.9 调研发现对设计的影响

| # | 发现 | 来源 | 对设计的影响 | 涉及 |
|---|---|---|---|---|
| 1 | 生产引擎把 batch invariance 和确定性采样做成一等特性，并用"批量对单条、token 和 logprob 完全一致"的测试守护 | DeepInfer | 参照模式在生产规模可行；但它会限制某些策略（MTP 变长验证必须关），而且需要按平台补齐 | D10 |
| 2 | 容差门禁会被绕过：省略约 39% 词表行的候选通过了 ppl/KL/argmax 门禁；还有去掉 EOS、只测到 prefill 路径、缓存失效 | SEIS | 参照模式要做 token 级和全词表 logprob 比对；评测要覆盖 decode 路径；停止行为必须一致；工作负载要包含变化的请求序列；定期代码审计 | D10、10.4 |
| 3 | 状态异构是现实：两个池（按 token 分页的 block KV 和每序列固定的 slot）、在 128 token 边界打快照、跨状态种类联合决定命中长度、近似重建 | DeepInfer | 状态种类插件需要增加"命中规则"和"重建方式"；近似重建会影响输出，必须声明 | D4、D10 |
| 4 | 生产里选择"按版本失效"，而不是"迁移格式" | DeepInfer Q&A | 版本化失效必选，迁移函数可选；兼容键进入系统身份 | D4、D12 |
| 5 | 大量决策已经是代价模型驱动的策略：chunk 由二次代价模型按时间预算反解；验证长度按接受率和实测吞吐曲线选取；EPLB 按窗口统计周期性重排；DP 负载均衡分档 | DeepInfer | 直接映射为 D7 的决策点和 D8 的代价模型；也是首批 policy 搜索的强基线 | D7、D8、10.6 |
| 6 | 策略收益随负载变化：DSpark 在 16 并发时 −1.5%，96 并发时 +11% | DeepInfer Q&A | 策略必须在工作负载分布上评估，不能只看单一工作点 | D7、10.3 |
| 7 | 异步调度下策略看到的状态晚一步；有些决策被下放到 GPU kernel | DeepInfer | 决策点接口要支持"CPU 定预算、GPU 定细节"的两段式；快照的时效要写进契约 | D7 |
| 8 | 派生产物缓存的键不完整：decode 吞吐曲线的键不含 kernel 版本，而它直接驱动调度决策 | DeepInfer | 派生产物（代价曲线、plan）必须挂在依赖图上，由 kernel 替换触发失效 | D3、D5 |
| 9 | 配置面靠 59 个环境变量；有变量绕过集中定义、用不同规则解析；mini-sglang 解析失败时静默回落默认值 | DeepInfer、mini-sglang | 所有配置都要声明、严格解析、只有一条读取路径、全部进入身份 hash；静态检查禁止裸 `getenv` | D12、原则 4 |
| 10 | 和基础设施耦合的部分仍然开不了源（KV cache 系统、负载均衡） | DeepInfer（2025-04 和 2026-10 两次表述） | D14 "内核里不允许出现私有依赖"的约束是对的；KV 存储和路由要定义成插件接口 | D14 |
| 11 | 平台抽象很薄，kernel 在导入时按平台整体分发，没有 shape 级路由、fallback 和运行时换绑 | DeepInfer | "definition × 硬件"矩阵在实践中已经存在；RSI 需要的增量是 shape 级路由、fallback、shadow、运行时换绑，以及显式的能力标志 | D6、D14 |
| 12 | mini 引擎可读但没有契约：全局可变状态、属性注入 KV、组件自己 capture graph；已停更约半年 | nano-vllm、mini-sglang | 证实"可读性是局部性质，不变量要靠结构强制"；mini 引擎适合作 sandbox，不适合作底座 | 4.1、D3 |
| 13 | 整引擎进化可行且有叠加效应，但继承的报告会携带错误结论 | SEIS | 支持把"叠加"作为一等目标；报告里的结论要链接到测量记录；迁移率要单独度量 | D12、13.2 |

---

## 4. 方案选择记录

### 4.1 不采用的方案（简述理由）

- **基于 vLLM 或 SGLang 做大规模重构的 fork**：upstream 最近 30 天分别合入了 1,928 和 2,283 个 PR，大重构的 fork 几周内就无法 rebase。DeepSeek 在 2025-04 自述过内部 vLLM fork 分叉后难以开源、难以维护；2026-10 开源的 DeepInfer 也只覆盖一个模型，和基础设施耦合的部分仍未开源（3.7）。
- **不 fork，只在现有引擎上加一层，加上向 upstream 提的接缝 PR**：能力上限取决于社区接受多少 PR，而且拿不到全局不变量。不作为主路线；可以在验证阶段作为对照组（见 13.2）。
- **基于 DeepInfer 做**：它是生产级的，但只支持一个模型、只有离线入口、外部 PR 基本不合入，也没有单元、契约和运行时变更机制。作为 DeepSeek 系列的参照实现和对照基准，比作为底座更合适。
- **整套重做一个全功能引擎**：需要的是一个新内核，不是复刻 vLLM 或 SGLang 的全部功能。kernel、模型实现、通信和传输都应该作为零件移植或复用。
- **追求所有层 Vite 式的热替换**：编译产物、native 扩展、拓扑变化在物理上做不到亚秒级。只在逻辑和参数层追求热替换，其余走温档或冷档，并诚实地返回结果状态。
- **直接拿 mini-sglang 或 nano-vllm 当产品**：它们读得懂，适合当参照和 sandbox（SEIS 就是在 mini-sglang 上做的），但没有按 RSI 契约设计，2026-08 以来也都没有新提交（3.8）。

### 4.2 采用的方案

**新的 RSI-native 内核，加上复用和移植过来的零件，加上成熟开源引擎作为常设对照。**

- 内核负责全局不变量：单元模型、状态服务、plan 编译器、调度机制、事务、溯源、自省、执行层抽象。
- 零件负责性能和覆盖面：kernel 库、通信和传输库、模型实现、API 层的解析器。
- 对照基准负责防止"在一个弱基线上自我感动"：所有正确性和性能结论都要对照 vLLM/SGLang；DeepSeek 系列另外对照 DeepInfer。

---

## 5. 设计动机：从 RSI 回路倒推

### 5.1 RSI 回路需要的性质

| 性质 | RSI 为什么需要它 | 局部还是全局 |
|---|---|---|
| 可读的单元和契约 | agent 只能在看得懂的范围里修改。SGLang 约 30 万行 Python，mini-sglang 约 5 千行，nano-vllm 约 1.2 千行；DeepInfer 约 5 万行，但只支持一个模型 | 局部 |
| 完整的失效追踪 | 否则热替换会静默不生效，或者读到旧状态（参见 vLLM #61287、SEIS 案例 3） | **全局** |
| 明确的状态归属和迁移 | 逻辑才能自由替换，状态不丢 | **全局** |
| 参照模式（确定性） | 调度和 kernel 改动后能做差分正确性比较，不只看指标（DeepInfer 已把它做成生产特性） | **全局**：只要有一个非 batch-invariant 的算子，差分比较就失效 |
| 身份和溯源 | 结果可重放、可审计、可叠加 | **全局**：一个绕过配置系统的环境变量就能破坏它（DeepInfer 的 C++ 调度器就自己读 `DEEPINFER_USE_NEXTN_DECODE`） |
| 跨 rank 的事务式变更 | 多卡下的安全替换 | **全局** |
| 多保真度评测 | 内循环快，同时结论可信 | 局部，前提是执行层可替换 |
| 评测器隔离 | 防 reward hacking（SEIS 的四个案例） | 局部 |
| 稳定的自省接口 | agent 和外部诊断工具拿到的证据可靠 | 半全局 |

### 5.2 为什么全局不变量要求新内核

- 全局保证只要有一处违反，就等于没有。在一个大而快速演进的代码库里维持全局不变量，需要审计全部代码，并且永远盯着新进来的提交。
- 新内核可以**靠结构强制**：
  - 组件不能自己 capture graph，只能由内核统一编译执行计划；
  - 所有配置读取和副作用都必须经由 context；
  - 状态只能通过状态服务访问。
- mini 引擎也没有解决这个问题。精简让代码读得懂，但不变量仍然是隐式的，SEIS 里的 agent 照样会踩进去（3.8）。

### 5.3 来自公开研究的设计依据

1. **接缝放在哪里，用数据决定。** Herschel 论文表 1 统计的历史 PR/CR 数量：
   - Graph capture/replay：1,954
   - 数据搬运与远端 KV：1,826
   - 通信与放置：1,798
   - 算子数据流：1,461
   - shape 相关的效率：1,389
   - KV 与运行时内存：1,146
   - Host 提交与调度：1,090
   - 状态复用：886
   - 投机解码与无用工作：375
   - 同步：276
   - 实现选择：195

   改动最频繁的几类，应该成为一等的单元和决策点。
2. **很多框架级改进是横切的。** Herschel 9.2 节里，agent 把 DP 元数据同步从 Worker 挪到 EngineCore，运行时代码改了 +766/−150 行；移植回 upstream 后，最终运行时 +707/−327、测试 +692/−16。SEIS 的最好引擎也是"重新设计整个引擎"的结果。这类改动没有任何插件边界能装下。所以对引擎层来说，**e2e 评测是主路径，热替换是次要的**；热和温路径主要服务于 kernel、参数和策略单元。
3. **算子级的提升不等于 e2e 的提升。** Herschel 的经验：
   - 算子计算快，准备和结果合并阶段可能很贵；
   - 去掉开销不保证带来收益；
   - 实验必须保留到达模式、shape 和依赖关系。
4. **内核要提供稳定的自省接口。** Herschel 的 Hprof 读取器 6 个月发了 5 个版本，就是在追引擎内部的布局变化。
5. **KREX 的思路可以推广到 e2e。** e2e 评测里的"区间外"主要是启动、编译和 capture。温路径可以把这部分压到接近零，区间独占计时负责测量窗口。另外，计时区间必须由评测器标注，不能交给 agent。
6. **生产引擎给出了现成的策略清单。** DeepInfer 把 chunk 预算、验证长度、EPLB 的窗口和周期、graph 分桶都写成了固定逻辑加环境变量参数（3.7）。这说明这些决策点在生产中真实存在，并且值得调。
7. **评测会被绕过。** SEIS 的四个案例（3.8）说明，e2e 门禁只要有一处没覆盖，agent 就会在那里拿正确性换速度。

---

## 6. 核心概念

### 6.1 接缝的五个属性

接缝指：不改那处代码，也能改变那处行为的位置（借自 Feathers）。一个合格的接缝必须明确以下五点，内核才能自动判断档位并验证改动生效：

| 属性 | 要回答的问题 |
|---|---|
| 接口 | 可替换的东西长什么样：参数和取值范围、definition、策略的方法签名 |
| 绑定点 | 引擎在哪里、什么时候去读"当前用哪个实现" |
| 生效时机 | 每次调用、每个 step、每个 plan，还是启动时一次 |
| 失效范围 | 改完之后要重建什么：什么都不用、清 KV、recapture、重新编译，还是重启 |
| 状态归属 | 能不能碰权重、KV、通信组；布局变了谁负责迁移 |

### 6.2 变更档位和结果状态

| 档位 | 耗时量级 | 保留什么 | 典型对象 |
|---|---|---|---|
| 热 | 毫秒 | 进程和全部状态 | 参数、策略单元、graph 外的 kernel |
| 温 | 秒到几十秒 | 进程和权重；派生产物（plan、graph、编译缓存、性能曲线）在后台重建，到 epoch 边界提交 | graph 内的 kernel、融合组、调度机制（调度宿主重启）、DP/EP 扩缩 |
| 冷 | 分钟级（可以用 CRIU 或 cuda-checkpoint 快照来加速） | 只保留缓存 | native 扩展、显存规划、新模型、TP 切换（在通用重分布机制完成之前） |

每次变更都返回以下三种结果之一：
- `applied`
- `restart-required`：附带原因和所需档位
- `failed`：附带证据，并已回滚

### 6.3 契约包络和包络失效日志

- **包络内的改动**是"注册一个新单元"，agent 能做得又快又安全。
- **包络外的改动**需要修改契约本身，也就是内核在演进，必须有人评审。
- 从第一天起记录**包络失效日志**：每次事件需要修改契约或内核时，记下事件类型、原因、改了哪个契约。这是衡量架构到底有多 RSI-native 的直接指标，也用来决定下一步加固哪个契约。

---

## 7. 设计原则

1. **机制与策略分离。** 正确性相关的操作放在机制层。所有"怎么选"的决定都由策略单元提出，由机制层校验。策略无法破坏不变量。
2. **状态比逻辑活得长。** 权重、KV、通信组、编译产物由长寿命的状态服务持有，带布局版本和失效规则（必要时带迁移函数）。逻辑单元尽量无状态。
3. **组合在编译执行计划时是动态的，执行时是静态的。** 契约、路由、依赖解析都在 plan 编译阶段完成，热路径上零间接调用。
4. **凡是影响输出或性能的东西，都要声明、都要记日志。** 没声明的依赖和配置读取，会破坏失效追踪和溯源。
5. **评测器在候选的信任边界之外。** 计时区间、参照输出、评分分母都由评测器持有，agent 看不到也改不了。

---

## 8. 架构总览

```
                 ┌──────────────────────── 控制面（常驻） ─────────────────────────┐
 agent / 人 ───► │ 单元注册表 · 契约 · 变更事务管理 · 溯源日志 · 自省 API · 权限      │
                 └───────────────┬──────────────────────────────────┬────────────┘
                                 │ 变更提案 / epoch 提交              │ 评测请求
                 ┌───────────────▼──────────────┐     ┌──────────────▼──────────────┐
                 │ 调度宿主（可独立重启）         │     │ 评测平面（独立信任域）        │
                 │ 调度机制 + 策略单元           │     │ trace 回放 · 门禁 · 计时区间   │
                 └───────────────┬──────────────┘     └─────────────────────────────┘
                                 │ 共享内存 IPC（每个 step）
        ┌────────────────────────▼──────────────────────────────────┐
        │ worker（每卡 / 每 rank）：状态服务 + plan 执行               │
        │ 权重存储 · 异构状态池 · 通信组 · 编译产物库 · 双缓冲 plan     │
        └────────────────────────┬──────────────────────────────────┘
                                 │ 执行层：真 GPU | 模拟执行器 | shadow
```

从另一个角度可以把系统分成四个平面：
- **状态平面**：状态服务。
- **逻辑平面**：可热替换的单元。
- **执行平面**：编译好的执行计划。
- **评测平面**：独立的信任域。

---

## 9. 架构决策详解

每个决策按"选项、选择、动机、收益、挑战"组织。

### D1. 进程拓扑与控制面

- **选项**
  - A：每个 rank 一个进程，SPMD（nano-vllm 接近这种：rank 0 和引擎同进程，其他 rank 由共享内存驱动）。
  - B：一个引擎核心进程加若干 worker（vLLM v1、SGLang、mini-sglang 的做法；DeepInfer 是每个 DP rank 一个引擎进程，调度器和 driver worker 同进程）。
  - C：三层结构：控制面、调度宿主、worker。
- **选择：C。**
- **动机**
  - 变更事务需要唯一的协调者。
  - 评测器需要独立的信任域。
  - 调度逻辑要能替换，同时不丢失 GPU 上的状态。verl 的教训是：worker 和 driver 同生共死，driver 一退出就要全部重来。
- **收益**
  - 调度宿主重启时，权重、KV、plan 都还在，机制级改动降为温档。
  - 控制面常驻，溯源连续不断。
- **挑战**
  - 每个 step 都要跨进程通信，必须走共享内存并和计算重叠。Herschel 9.1 节的案例正是 Worker 在 dequeue 上等待、GPU 空转。
  - 调度宿主重启时在途请求要能重建，需要把调度状态做成可快照、可重放、带版本的日志。
  - DeepInfer 的生产经验是：单引擎重启要分钟级，所以容错放在集群层（冗余配额加请求重路由）。这对线上服务够用，但对 RSI 内循环不够：重启成本仍然直接决定迭代速度。

### D2. 语言分层

- **选项**
  - 全部 Python（nano-vllm、mini-sglang、SGLang 的做法，靠重叠调度掩盖 CPU 开销）；
  - 机制核心下沉到 C++，但策略也写死在里面（DeepInfer：调度器、block manager、输入准备约 5.9 千行 C++，其余约 4.6 万行 Python）；
  - 分层：机制用编译语言，策略用 Python。
- **选择：分层。** 机制层用 Rust 或 C++（具体选哪个待定）；策略单元和模型定义用 Python。策略接口传递的是批量化、向量化的数据结构。
- **动机**
  - 机制要稳定且快；策略层是 agent 最常修改、最需要热替换的部分。
  - DeepInfer 说明调度和 block 管理下沉到 C++ 在生产规模上是值得的；但它的策略（chunk 预算、prefill 尝试间隔、验证长度）也写在 C++ 里，改一次就要重新编译。
- **收益**
  - 机制层的性能和正确性有保障。
  - 策略层对 agent 友好。
  - 稳定下来的策略以后可以再下沉成编译代码。
- **挑战**
  - FFI 边界上的调试和构建复杂度（例如 PyO3、pybind）。
  - agent 要同时面对两种语言。
  - 每次 Python 策略调用都要有 CPU 时间预算，并且能落在重叠窗口里。

### D3. 单元模型（组合语义）

- **选项**：类继承加注册表（现状，mini-sglang 也是这样）；Cordis 式的 context 加声明式依赖加可撤销 effect；actor 模型（Monarch、Ray）。
- **选择：采用 Cordis 语义，并加三项扩展。** 每个单元声明：
  - 接口、依赖的服务、所有副作用（必须经由 context 注册）；
  - **绑定时机**：每次调用、每个 step、每个 plan，还是启动时一次；
  - **所持状态**：自有还是借用，带布局版本和失效规则；
  - **参照实现、不变量和契约测试**。

  另外，派生产物也算依赖：执行计划依赖于单元版本；实测的性能曲线依赖于 kernel solution 和 plan。
- **动机**：让失效集合可计算、档位可自动判定；让 agent 面对的是一份契约目录，而不是全部代码。
- **收益**：热、温、冷档位可以自动判定；回滚等于撤销 effect；单元目录本身就是给 agent 的文档。
- **挑战**
  - Python 不强制 effect 追踪，需要静态检查（禁止模块级可变全局状态、禁止私自 capture graph、禁止裸 `getenv`）和运行时守卫。
    - 反面例子：nano-vllm 用模块级可变的 `_CONTEXT` 传 attention 元数据，用 `setattr` 把 KV 注入各层；DeepInfer 的 `DecodePerfCache` 换 kernel 后仍会沿用旧曲线（键里没有 kernel 版本）。
    - 正面例子：DeepInfer 用 `forbid_cuda_sync` 在热路径上做运行时守卫；mini-sglang 的全局 `Context` 只允许设置一次。
  - 对贡献者有纪律要求。
  - 单元的粒度怎么划。

### D4. 状态模型：异构状态池（最关键的决策）

- **选项**：先以 paged KV 为中心，之后给新模型补"混合分配器"（现有引擎走的路）；从第一天起就把"每层的状态规格"作为一等抽象。
- **选择：后者。** 每一层声明自己的状态规格：
  - 状态粒度：按 token、按 chunk 还是按请求；
  - page 大小；
  - 能否做前缀复用，以及如何打快照；
  - 能否在 P/D 之间传输；
  - 淘汰语义。

  需要覆盖的状态种类：
  - 普通 paged KV；
  - 压缩 KV（DeepSeek V4 的 CSA/HCA，压缩比 4 和 128）和 indexer 的 key pool；
  - SWA 环形缓冲区和压缩边界累加器（DeepSeek-V4.1-Flash，每序列一个固定 slot）；
  - recurrent state 和 conv state（Qwen3.5 的 Gated DeltaNet）；
  - MTP drafter 的 KV；
  - 多模态 encoder 的缓存。

  状态种类本身也是插件，每种必须实现：分配、按前缀切片、快照、传输、失效，以及可选的迁移。

  权重存储同样是状态服务：按逻辑名寻址、带版本、支持原地 refit。量化和切分方式用带版本的**布局描述符**表示，描述符要能把每个张量描述成"逻辑坐标到分片"的映射，为 D9 的通用重分布做准备。
- **DeepInfer 对 V4.1-Flash 的实现可以当作一次真实的包络测试**（3.7）：
  - 两种粒度并存：按 token 分页的 block KV，以及每序列一个固定 slot 的 SWA 缓冲区和压缩累加器；分配必须联合进行。
  - 快照只在 128 token 边界打；decode 期间跨边界时在显存里拷副本。
  - 前缀命中长度要跨状态种类联合决定：由 block KV 定长度，SWA 缺失时回退一个块重算。
  - 有近似重建：bounded replay 只重算 128 个 token，得到近似的 SWA KV，结果会随缓存命中与否而不同。
  - 生产里选择按版本失效，而不是迁移格式。

  据此给状态种类插件的接口补三条：
  - 除了上面的操作，还要声明**命中规则**（自己的对齐粒度，以及缺失时怎么回退）和**重建方式**（精确重算还是近似重建；近似重建要声明为"影响输出"）。
  - **版本化失效是必选的，迁移函数是可选的。** 兼容键（模型权重、布局版本、tokenizer、影响数值的单元版本）进入 D12 的系统身份。
  - 联合准入：任一种状态分配不出来，整个请求就不准入。
- **动机**：每一代 Qwen 和 DeepSeek 都会带来新的状态类型，这是"只支持两个模型家族"范围下仍然最大的变量。
- **收益**
  - 新模型家族只需声明状态规格。
  - 淘汰、offload、复用策略都变成可进化的单元。
  - 布局变化时走失效或迁移，而不是重启。
  - 这是从零构建相对现有引擎的真实优势之一：现有引擎都是后来补的混合分配器。
- **挑战**
  - 抽象要足够通用，这是整个设计里最难的部分。
  - recurrent state 和 SWA 做前缀复用，需要在 block 边界打快照。
  - 碎片化。
  - 异构状态在 P/D 之间传输。
  - 布局一变，kernel 和 plan 都要失效。
  - 近似重建和参照模式冲突（见 D10）。

### D5. 执行模型与 plan 编译器

- **选项**
  - A：eager 执行，由引擎 capture decode graph（SGLang、mini-sglang、nano-vllm、DeepInfer 的做法）。
  - B：torch.compile 加 piecewise graph（vLLM 的做法）。
  - C：自建 plan 编译器，统一负责所有 capture，torch.compile 只作为可选的融合后端。
- **选择：C。**
  - 组件不能自己 capture graph。
  - 每个 plan 记录自己依赖的单元版本、kernel solution id、参数取值和状态布局。
  - 采用**双缓冲 plan**：新 plan 在后台的独立 stream 和内存池里准备，到 epoch 边界再提交。
  - **融合组是显式的单元。** 被 Inductor 自动融合掉的算子，agent 写的 kernel 就替换不进去了。Herschel 表 1 里"算子数据流"一类有 1,461 个历史 PR，说明融合本身就值得进化。
  - 由 plan 派生出的性能数据（例如每个 graph 尺寸的实测耗时）也是派生产物，挂在同一张依赖图上，随 plan 一起失效。
- **动机**：让温档变更接近零停机（Elastic EP 的 prepare/commit 做成通用机制）；让失效范围精确到具体的 plan。
- **收益**：只重建受影响的 plan；融合决策可以被进化；plan 的依赖集合就是溯源信息的一部分。
- **挑战**
  - 双缓冲 plan 占用额外显存。
  - capture 耗时。
  - 动态 shape 的分桶策略（这本身也应该是一个策略单元）。现有引擎的分桶各不相同：DeepInfer 先按 2 的幂、再每 8 一档；nano-vllm 是 1、2、4、8 之后每 16 一档；mini-sglang 是 1、2、4 之后每 8 一档。
  - kernel 必须可以被 graph capture。
  - 和 Inductor 集成的复杂度。
  - 平台能力差异（见 D14）。

### D6. kernel 绑定：definition 注册表

- **选项**：直接调用；torch.library 的分发机制；按平台整体切换实现目录（DeepInfer 的做法）；按 definition 和 shape key 路由的注册表。
- **选择：注册表。**
  - **definition**：参照语义（`reference.py`）、固定轴和可变轴、dtype、布局、容差、是否 batch-invariant。
  - **solution**：目标架构、DSL、作者（人或 agent）、来源、依赖了哪些库、workspace 需求、能否被 graph capture。
  - 路由在 plan 编译时解析进 plan，graph 内零开销；eager 路径逐次查表，微秒级开销。
  - 永远有 fallback；支持 shadow 和 canary 绑定。
  - 注册表是"definition × 硬件架构"的矩阵，每个 definition 都必须有一个可移植的兜底 solution（PyTorch、Triton 或 TileLang）。
- **现有引擎里的雏形**：
  - DeepInfer 每个 kernel 一个接口模块加测试，在导入时按平台整体选 `cuda/` 或 `asc/` 下的实现，缺失时抛 `KernelUnavailableError`。
  - mini-sglang 用注册表选 attention 后端，prefill 和 decode 可以选不同后端。
  - 它们缺的是 RSI 需要的部分：按 shape 路由、fallback、shadow 和 canary、运行时换绑。
- **动机**：打通 kernel agent 到引擎的路径；把算子级 benchmark 的评测契约（Atrex-Bench 的五文件格式、flashinfer-bench 的 Definition/Solution）升级为部署契约。
- **收益**
  - 链路打通：生产 profiling 发现某个 shape 上的算子 SOL 偏低 → 自动生成 benchmark 题目 → kernel agent 优化 → 注册 → e2e 评测 → canary。
  - 可以按 shape 做专门化。
  - 新硬件的 kernel 覆盖可以按矩阵批量推进。
- **挑战**
  - definition 的粒度：取融合算子还是基础算子？
  - 布局约定随时间漂移。
  - 容差策略怎么定。
  - definition 数量膨胀，需要评审和归一化。
  - 新 solution 的 kernel 数量和旧的不同，会改变 graph 拓扑，必须 recapture。

### D7. 调度：机制核心加决策点

- **选项**：单体 scheduler；在单体上加 hook（vLLM RFC #51608 的方向）；机制核心加纯函数式的策略决策点。
- **选择：最后一种。**
  - 机制核心持有正确性关键的操作：KV 分配、请求状态机、batch 是否合法、抢占的执行。
  - 决策点包括：准入、排序、chunk 大小、抢占对象、draft 长度和验证长度、graph 分桶、缓存淘汰、路由、EPLB 的统计窗口和重排周期。
  - 策略的输入是只读快照，输出是决定，由机制层校验。
  - 允许策略声明跨 step 的状态，并提供迁移函数。
- **生产参照**（DeepInfer，3.7）：
  - prefill chunk 由二次代价模型按时间预算反解，在 DP 组内取最小；
  - 验证长度按接受率和实测吞吐曲线，选边际收益最大的前缀；
  - EPLB 的冗余数、统计窗口和重排周期；
  - 这些都是现成的强基线。DSpark 的收益随并发从 −1.5% 变到 +11%，说明策略必须在工作负载分布上评估，最好能随负载自适应。
- **动机**：policy 层的 RSI 是目前的空白区；纯函数接口让策略可以热替换、可以在模拟器上跑、可以做反事实评估。
- **收益**
  - 策略可以在 epoch 边界热替换。
  - 可以在模拟执行器上运行。
  - 可以对离线快照做反事实评估和 A/B。
  - 正确性由机制层保证。
  - Herschel 表 1 里"Host 提交与调度"、"Graph capture"、"投机解码"这几类，可以直接映射成策略单元。
- **挑战**
  - 决策点的粒度怎么定。
  - 每个 step 构造快照的 CPU 成本。
  - 多个策略之间会耦合，比如 chunk 大小和 graph 分桶。
  - 异步调度下，策略看到的快照比实际状态晚一步（DeepInfer 调度 N+1 时拿不到 N 的结果）。契约里要写明快照的时效。
  - 有些决策的输入只在 GPU 上。DeepInfer 让 CPU 只定验证总预算，每个请求的长度由 GPU kernel 计算。决策点接口要支持这种"CPU 定预算、GPU 定细节"的两段式；GPU 那一段本身就是一个 kernel solution，走 D6 的契约。
  - 新的调度范式（新的决策点、新的请求状态机）属于机制改动，只能走温档或冷档。

### D8. 执行层抽象与模拟

> 2026-11-04：v0 不写模拟执行器，见第 15 节第 5 条。本节保留，作为长期方向的记录。

- **选项**：只有真实执行；独立重写一个模拟器（Vidur 式）；同一份代码，执行层可以替换。
- **选择：可替换的执行层。**
  - 真 GPU 执行器；
  - 模拟执行器：代价模型按算子、shape、硬件，用生产 profile 校准；
  - shadow 模式；
  - 外加一个确定性的虚拟时钟。
- **动机**
  - ADRS 说明了模拟器对进化的关键作用；但独立的模拟器和真系统是两套实现，会产生偏差。
  - 生产引擎已经在用代价模型：DeepInfer 有 prefill 的二次代价模型、按请求的 FLOPs 估算器和实测的 decode 吞吐曲线（3.7）。模拟执行器可以从这些现成的模型起步。
- **收益**
  - 策略和机制的评估从分钟级降到秒级，可以在 CPU 上大规模并行。
  - 跑的是真实的调度和 KV 代码。
  - 第一阶段不需要 GPU 就能开始做 policy 层的 RSI。
- **挑战**
  - 代价模型的保真度：资源争用、通信重叠、CPU 开销、graph padding 都要建模。
  - 校准要持续更新，而且要随 kernel 替换失效（D3）。
  - 多实例和 P/D 场景的模拟更复杂。
  - 模拟结论必须在真机上复核。

### D9. 分布式与变更事务

- **复用**：torch.distributed/NCCL（昇腾上是 HCCL）、DeepEP、Mooncake/NIXL。
- **自建**
  - 通信组作为带版本的状态服务。
  - **step epoch 协议**：每个 step 有一个 epoch 号，所有变更只在 epoch 边界提交。提交走两阶段：所有 rank 先 prepare，过 barrier 后 commit，任一 rank 失败就整体 abort。
  - 基于 D4 布局描述符的**通用 M 对 N 重分布规划器**（DynaTrain 的 VPS、Tenplex 的 PTC 都是这个思路），用于 EP、DP 以及以后的 TP 切换。
- **动机**
  - 多卡下热和温变更的安全性；弹性重配置复用同一套机制。
  - "在 step 边界改权重布局"在生产里已经是常规操作：DeepInfer 的 EPLB 每隔若干 step 在线重排专家权重。但它的 EP 故障恢复仍然是重启，再按健康卡数重定 EP 大小。
- **收益**：安全的分布式变更；切换机制不必为每种组合手写。
- **挑战**
  - NCCL communicator 创建慢，而且有死锁风险。
  - 部分失败的处理。
  - 和在途的异步操作交错。
  - 切换时需要显存余量。
  - P/D 跨实例需要跨实例事务。

### D10. 参照模式（确定性）

- **选项**：不做；只做到容差级的比较；一等公民的参照模式。
- **选择：一等公民的参照模式。**
  - batch-invariant 的 kernel 加按请求固定种子的采样；definition 带一个"batch-invariant"标记。
  - 参照模式下比对 token 序列和全词表 logprob（至少是 top-k logprob 加整体分布的统计量），不只比 argmax 一致率。
  - 生产模式仍然可以用非确定性的快 kernel，用容差检查。
- **动机**
  - 调度和引擎改动本来就不应该改变输出，token 级完全一致是最强的门禁。
  - 生产引擎已经这么做了：DeepInfer 把 batch-invariant 推理和确定性采样做成一等特性，用"批量对单条、token 和 logprob 完全一致"的测试守护（3.7）。
  - 只用容差门禁会被绕过：SEIS 里省略约 39% 词表行的候选通过了 ppl/KL/argmax 门禁（3.8）。
- **收益**：防刷分能力强；正确性检查比跑质量评测快得多。
- **挑战**
  - 每个算子都需要一个 batch-invariant 实现，包括 split-K GEMM、MoE。
  - all-reduce 的归约顺序也要固定。
  - 参照模式本身有性能代价。
  - 新模型和新硬件都要补齐这一列（DeepInfer 的参照模式目前只在 NVIDIA 上可用）。
  - 有些策略和参照模式不兼容：DeepInfer 的确定性采样要求关闭 MTP 变长验证。这类策略只能走容差模式加质量评测，需要在单元上声明。
  - 近似重建（如 SWA bounded replay）让输出依赖缓存的命中状态。参照模式要么关掉近似重建，要么把缓存状态固定为评测输入的一部分。

### D11. 自省与可观测性

- **选择**：引擎内置以下能力：
  - 带版本、稳定的自省 schema：调度状态、KV 状态、plan 状态、算子调用（含 shape key）；
  - 按单元归因的耗时；
  - 按需打开的深度 trace 钩子，和外部 profiling 工具（例如 Herschel 类系统）兼容。
- **动机**：外部工具的读取器不用再追着内部布局改；agent 拿到可靠的证据。
- **收益**：profile 可以直接映射到 definition 和单元；kernel benchmark 的题目可以自动生成。
- **挑战**：运行开销和数据量；schema 既要稳定又要能演进。

### D12. 身份、溯源与信任边界

> 2026-11-04：v0 只做身份和溯源，权限部分推迟，见第 15 节第 7 条。

- **选择**
  - **系统身份**是一个 hash，覆盖：单元图、各单元版本、kernel solution、参数、状态布局、硬件和驱动版本。每条测量数据都带上这个身份。
  - **每次变更都是一条事件**：改了什么、为什么改、证据、档位、结果、审批人。产物按内容寻址。即 experiment-visible means logged。
  - **权限**：agent 按单元类型授权可以改什么。评测器的修改单独审批。
  - **配置**：所有配置都声明、严格解析、只有一条读取路径、全部进入身份 hash；静态检查禁止裸 `getenv`。
  - **报告溯源**：agent 报告里的结论要链接到产生它的测量记录。
- **动机**
  - 可重放、可审计、可叠加；沉淀 agent 轨迹，作为将来研究用的数据。
  - 配置蔓延的现实样本：DeepInfer 有 59 个集中定义的环境变量，首次读取后缓存；C++ 调度器另外自己读其中一个，用不同的解析规则；部分默认值在运行时写回 `os.environ`。mini-sglang 的环境变量解析失败时会静默回落到默认值。
  - SEIS 发现继承下来的报告会把错误结论一起传下去。
- **收益**：任何结果都能追溯到确切的系统状态；失败的尝试也保留下来，方便定位问题（Herschel 提到失败案例评审很慢）。
- **挑战**：配置蔓延；环境变量和依赖库版本必须全部固定并纳入 hash；测量本身有噪声。

### D13. 模型定义方式

- **选项**：每个模型手写一个文件（现状）；从 HF config 通用生成；声明式模型规格。
- **选择：声明式模型规格。** 由已注册的 definition、状态规格和融合组组成。vLLM、SGLang 和 DeepInfer 的模型文件移植成初始 solution。
- **动机**：plan 编译器能看到完整的模型图；融合和顺序可以被进化；新模型等于"新规格加新 definition"。
- **收益**：新模型接入的工作对 agent 友好，因为 HF 的 modeling 代码就是对照实现。
- **挑战**：表达力，比如 mHC hyper-connection、CSA/HCA、YOCO 的 encoder/decoder 分层、Engram、MTP；权重映射；调试体验。

### D14. 平台抽象与开源边界

> 2026-11-04：v0 不开源，但"内核里不允许出现私有依赖"的约束保留，见第 15 节第 10 条。

- **选择**
  - 平台插件负责：设备、分配器、stream 和 graph、通信后端、profiler 钩子、计时基础设施支持。
  - plan 编译器按平台读取**能力标志**（能不能 capture graph、有没有 conditional node、内存池语义、有没有 batch-invariant 实现），做不到的就降级。
  - 部署方私有的集群管理、存储和加速器全部做成插件，内核里不允许出现私有依赖。KV cache 存储和请求路由也定义成插件接口。
- **动机**
  - 同时支持 NVIDIA 和其他加速器。
  - 避开"和部署基础设施紧耦合、没法开源"的坑。DeepSeek 两次遇到这个问题：2025-04 因此没有开源内部引擎；2026-10 开源 DeepInfer 时，KV cache 系统和负载均衡仍因同样的原因没有开源。
- **参照**：DeepInfer 的平台层很薄（3.7）：一个 `Platform` 类，加上按平台整体切换的 kernel 目录；NPU 路径靠 `transfer_to_npu` 把 `torch.cuda.*` 映射过去，再在初始化时设环境变量、打 monkey patch。这足以跑通一个模型，但平台之间的能力差异（例如 batch invariance 只在 NVIDIA 上有）没有被显式声明。
- **收益**：从第一天起可以开源；新硬件的接入有明确的工作面。
- **挑战**：插件接口要长期稳定；部署方私有插件的兼容性测试。

---

## 10. e2e 集成评测平面

### 10.1 为什么先做它

- 它是连接 kernel benchmark、kernel agent、计时基础设施和生产诊断工具的缺失环节，也是 policy 层 RSI 的前提。
- 它和引擎无关，对 vLLM、SGLang、DeepInfer 和新内核都能用。
- 它产出的数据会反过来指导内核哪些部分优先加固，比如局部改动和横切改动的比例、一轮迭代里各阶段的耗时占比。

### 10.2 契约（沿用算子级 benchmark 的格式）

| 内容 | 对 agent 可见？ |
|---|---|
| 工作负载描述：从生产 trace（公开 trace 或部署方自带）得到的到达过程、输入输出长度分布、前缀共享结构 | 可见 |
| 模型和部署配置、SLO | 可见 |
| 对照引擎的基线数字 | 隐藏 |
| 参照引擎的输出（logprob 或 token 序列） | 隐藏 |
| 评分分母和 roofline 式的下界 | 隐藏 |
| 部署权重（按 GPU 小时计） | 作为发布级别的产物单独维护 |

### 10.3 评分

- SLO 约束下的 goodput，以及在 SLO 下能达到的最大吞吐。
- 评分要覆盖负载分布，而不是单一工作点（DSpark 的收益在不同并发下从负变正，见 3.7）。
- 可以再加一个由逐算子 SOL 沿关键路径汇总出来的模型级下界，做 roofline 式归一化。这一点是否足够准还需要验证。
- 跨部署聚合时，按 GPU 小时加权（和 Atrex-Bench 的重要性权重、Herschel 的 Hotness 是同一个思路）。

### 10.4 防刷分

- 输出长度和停止行为必须一致；用短生成探针检查正常终止（SEIS 案例 2）。
- 参照模式下做 token 级和全词表 logprob 比对；生产模式下按容差比较，再加一个以生成为主的质量评测，不能只靠多选题的选项概率（SEIS 案例 1、4）。
- 工作负载必须包含变化的请求序列，不能只对固定 prompt 反复计时（SEIS 案例 3）。
- 监控前缀缓存命中率是否异常偏高，以及 warmup 带来的偏差。
- 记录每个 kernel solution 依赖了哪些库（对应 Atrex-Bench 指出的"正确性错觉"）。
- 计时区间由评测器标注（KREX 的教训）；评测器的修改单独审批。
- 定期做代码审计，任务分数不能替代审计（SEIS 案例 4）。评测器本身也要随引擎演进，但演进走单独的审批。

### 10.5 多保真度评测级联

> 2026-11-04：v0 的评测改为 patch 提交加冷启动，测试分层也改了，见第 15 节第 5、8 条和 v0-plan.md 的 2.5 节。

1. 单元级 microbench：算子级 benchmark 契约加区间独占计时，秒级。
2. 模拟执行器回放：秒级。
3. 真机回放，在受控规模下：分钟级。
4. shadow 或 canary：小时级，由部署方的评审和 canary 流程负责。

### 10.6 三条闭环

- **kernel 路径**：生产 profiling 发现某个 shape 上的算子 SOL 偏低 → 自动生成 benchmark 题目 → kernel agent 优化 → 注册进 D6 → e2e 评测 → canary。前提是 kernel agent 的输出契约增加一个"引擎可加载"的格式。
- **policy 路径**：用生产 trace 校准模拟执行器，在模拟器里搜索策略单元，再到真机复核。首批候选对象：
  - graph capture 和分桶策略；
  - 投机解码的 draft 数量和验证长度。目标函数可以直接用 Herschel 的结论："投机解码应该按有用输出来评判"。DeepInfer 的 DSpark 是现成的强基线；
  - chunk 大小和并发上限。DeepInfer 的"时间预算加二次代价模型"是现成的强基线；
  - cache 感知的路由；
  - EPLB 的统计窗口和重排周期。
- **引擎路径**：agent 产出的横切改动（Herschel Optimizer、SEIS 这类）走冷路径，统一用这一评测平面验收。被接受的修复沉淀为新的诊断模式和参考修复，失败的尝试也保留。

### 10.7 测量基础设施

> 2026-11-04：资源分配不归引擎管，评测一律冷启动，见第 15 节第 8、11 条。

- 把区间独占计时（KREX 的思路）扩展到多卡和多节点。
- 引擎的温路径加上 CRIU 或 cuda-checkpoint 快照，把 e2e 评测的"区间外"成本（启动、编译、capture）压到最低。

---

## 11. 压力测试：五类事件

| 事件 | 架构装得下吗 | 主要依赖的设计 | 生效档位 | 对 agent 的难度 | 人负责什么 |
|---|---|---|---|---|---|
| (5) 换用新的 GPU kernel | 装得下，这是架构优化的目标 | D6、D5 | graph 外热，graph 内温 | **容易** | 批准新的 definition |
| (2a) 新的调度策略 | 装得下 | D7、D8、D10 | 热 | **容易** | 基本不需要 |
| (1a) 新模型，在包络内 | 装得下 | D4、D6、D13 | 冷（新部署） | **中等偏易** | 批准新的 definition 和状态规格 |
| (4) 新硬件（如昇腾） | 装得下，但"复用现成库"的假设会部分失效 | D14、D6 | 冷 | 平台适配**难**；kernel 覆盖和调优**容易**，可以并行 | 平台插件、通信库、数值问题 |
| (2b) 新的调度机制 | 装得下 | D1、D7 | 温 | **中等** | 审查机制不变量 |
| (1b) 新模型，在包络外 | 需要演进内核 | D4 新增状态种类 | 冷 | **难** | 设计评审 |
| (3) 动态切换 TP/DP/EP | 装得下，但机制成本最高 | D4、D9、D5 | 温，秒级停顿 | 切换**策略**容易；切换**机制**很难，不交给 agent | 实现并验证切换机制 |

从易到难：(5) < (2a) < (1a) < (4) 的 kernel 部分 < (2b) < (4) 的平台部分 < (1b) < (3) 的 TP 切换。

### (5) 换用新的 GPU kernel

- **流程**：kernel agent 产出 solution → 隔离验证（算子级 benchmark 契约加区间独占计时）→ shadow 或 canary 绑定 → e2e 评测 → 后台重建受影响的 plan 和派生的性能曲线 → epoch 边界提交。回滚只需重新绑定。
- **难点**
  - 需要不同的数据布局：走 D4 的布局迁移或新建 definition。
  - 跨算子融合：新建融合组。
  - kernel 数量变化：必须 recapture。
  - 不能被 graph capture：只允许绑定在 eager 路径。
  - JIT 编译慢：放在 prepare 阶段提前编译。
  - 需要额外显存：在元数据里声明。
  - 计算和通信融合：会依赖通信组。
  - 依赖 kernel 性能的策略（例如按吞吐曲线选验证长度）要随之重新校准。
- **两个坑**：正确性错觉（用依赖记录来防）；算子快了端到端不一定快（用 e2e 结果把关）。

### (2) 新的调度模块

- **策略级（2a）**：纯函数接口；评测是模拟器几秒、真机回放几分钟，最后在参照模式下检查 token 完全一致；在 epoch 边界热替换；额外检查每个 step 的 CPU 时间预算。和参照模式不兼容的策略（例如变长验证）改走容差模式加质量评测。
- **机制级（2b）**：调度宿主单独重启，worker 上的状态都还在，所以是温档。在途请求要么排空、要么按版本化快照迁移。机制代码必须通过不变量测试（没有 KV 泄漏、没有重复分配、每个请求最终都能完成、公平性有界），适合用 property-based testing 来写；mini-sglang 空闲时的完整性检查（空闲页 + 缓存页 = 总页数）就是这类不变量的最小例子。模拟器同样能跑新机制。如果同时改了调度宿主和 worker 之间的协议（比如 Herschel 9.2 那种），按冷路径处理。

### (1) 新模型或新架构

- **包络内**：新的 head dim、RoPE 变体、MoE 规模、共享专家、MTP，以及 D4 已经覆盖的状态种类。agent 的流程：
  1. 读 HF 的 modeling 代码；
  2. 映射到已有 definition，或者新建 definition；
  3. 声明状态规格，写权重映射；
  4. 在参照模式下和 HF 或参照引擎（vLLM、SGLang、DeepInfer）对齐；
  5. 交给 kernel agent 优化新算子；
  6. e2e 评测。

  API 层的解析器用测试语料对齐。
- **包络外**：跨层共享 KV、非前缀语义的状态、改变序列位置的 token 合并或丢弃、访问外部 memory 的注意力等。需要往 D4 里新增状态种类，必须有人做设计评审。mHC 这类结构不影响状态层，但会冲击激活 buffer 规划和融合组。YOCO 式的跨层 KV 投影和 Engram 这类外部查表也要在包络测试里覆盖。
- **提前加固**：写代码之前，先用 Qwen3.5、DeepSeek V4/V4.1 和两三个假设架构，在纸面上把状态规格写完，作为"包络测试"。V4.1 的状态布局已经有 DeepInfer 的实现可以对照（D4）。

### (3) 动态切换 TP/DP/EP

- **DP 和 EP**：通信组换版本、迁移专家权重、重建 all2all buffer、后台准备新 plan、在 epoch 边界提交。Elastic EP 已证明可以做到 1–2 秒的停顿；DeepInfer 的 EPLB 也已经在生产里周期性地在线重排专家权重。
- **TP**：权重和 KV 都要重新切分，KV 也可以选择迁移或重算（PAT 用代价模型来决定）。还要新建通信组、重建全部 plan，并预留显存余量。P/D 两侧 TP 不同时，KV 传输要做布局转换。
- **原则**：
  - 机制由核心维护者实现并验证。
  - agent 进化的是切换策略：何时切、切成什么配置。
  - 很多场景下，集群层面增减实例再配合路由策略，成本更低，而且完全在包络内。DeepInfer 的生产部署也没有用 TP 和 CP，而是靠 DP/EP 加负载均衡。
- **顺序**：先做 DP/EP 弹性；TP 切换放到后面，或者先做"预置的两种布局之间切换"。

### (4) 新硬件：非 NVIDIA 加速器（以昇腾为例）

- **已核实的事实**（DeepInfer 源码和文档）：
  - DeepInfer 在 Ascend 950（CANN 9.2）上运行和 NVIDIA 相同的模型；
  - 平台抽象很薄：`Platform` 类加按平台整体切换的 TileLang kernel 目录（`cuda/` 和 `asc/` 各约 30 个文件，集合不完全相同）；
  - 通信用 HCCL；DeepEP 和 DeepGEMM 各有一个独立的 Ascend 移植（DeepEP-Ascend、DeepGEMM-Ascend）；
  - NPU 路径依赖 torch_npu 的 `transfer_to_npu` 把 `torch.cuda.*` 映射到 `torch.npu.*`，再加一批环境变量和 monkey patch；
  - batch-invariant 推理和确定性采样目前只支持 NVIDIA；Engram 的 CPU offload 在 NPU 上还不支持。
- **架构侧**：平台插件，加上 plan 编译器的能力标志；模拟器的代价模型用新硬件上的 profile 重新校准。
- **最大的坑**：FlashAttention-3、FlashMLA、DeepGEMM 依赖 Hopper/Blackwell 的专有指令；DeepEP 依赖 NVSHMEM/IBGDA。这些在其他加速器上不可用，要么逐个移植（DeepInfer 的做法），要么换成可移植实现。Triton、TileLang 这类多后端 DSL 写的 kernel 和 PyTorch 参照实现迁移成本最低。所以 D6 必须有可移植的兜底 solution，D14 必须显式声明能力差异。
- **agent 侧**：按重要性权重，批量把"definition × 新硬件"这一列交给 kernel agent，可以大规模并行。前提是 kernel agent 的知识库补上新硬件的规格，计时基础设施支持新硬件。
- **人负责**：平台插件打通、通信库集成、数值差异、graph 语义差异、和厂商协调。

---

## 12. 风险与缓解

| 风险 | 表现 | 缓解 |
|---|---|---|
| D4 状态抽象不够通用 | 每代新模型都撑破抽象，退化成"后补的混合分配器" | 先做纸面上的包络测试（V4.1 可以对照 DeepInfer 的实现）；状态种类插件化；记录包络失效日志 |
| 抽象开销吃掉性能 | 每个 step 的 CPU 时间上升，GPU 空转 | 原则 3：热路径零间接；给每个 step 的 CPU 时间设预算并持续监控；重叠调度 |
| 长期达不到对照基准 | 结论建立在弱基线上 | vLLM/SGLang（DeepSeek 系列加 DeepInfer）从第一天起挂在 CI 上作为对照；做迁移率检验 |
| 第二系统效应 | 过度设计，迟迟不能交付 | 严格按阶段推进；每个阶段都必须产出可用的 RSI 结果 |
| 评测被刷分 | agent 优化指标而不是系统本身 | 评测器隔离；参照模式；隐藏产物；输出一致性检查；变化的请求序列；定期代码审计（SEIS 的四个案例） |
| 模拟和真实之间的差距 | 模拟器上选出的策略在真机上无效 | 多保真度级联；持续校准；模拟结论必须真机复核 |
| 维护带宽和社区采用 | 新内核早期覆盖面小，用户没有理由迁移；mini 引擎发布后少有维护，DeepInfer 的外部 PR 难以合入 | 新内核先服务 agent 实验和研究用户；生产部署建议继续用成熟引擎，达到对照基准后再推广；从第一天起写清楚贡献流程和评审标准 |
| 开源版和私有部署分叉 | 部署方的私有插件和内核脱节，重演"开源版拿不出关键部件"的问题 | D14：内核不允许出现私有依赖；KV 存储、路由等定义成插件接口 |

---

## 13. 建设路线与验证

### 13.1 阶段

> 2026-11-04：本节的阶段划分已被 [v0-plan.md](v0-plan.md) 的里程碑 M0–M6 取代，保留在这里作为历史记录。

1. **第一阶段**：控制面骨架、单元和契约模型（D3）、溯源（D12）、评测接口、模拟执行器（D8）、调度的机制与策略拆分（D7）。e2e 评测平面的契约同步设计。这一阶段**不需要 GPU**，就能开始做 policy 层的 RSI，用来低成本地验证契约设计。
2. **第二阶段**：worker 的状态服务（D4）、plan 编译器（D5）、kernel 注册表（D6，接入 FlashInfer 和 kernel agent 的 solution）、参照模式（D10）、自省（D11）。先做一个模型、单节点，和 vLLM/SGLang 对齐；打通 kernel 路径的闭环。
3. **第三阶段**：MoE/EP（DeepEP）、DeepSeek 系列（对照 DeepInfer）、P/D 分离（Mooncake/NIXL）、跨 rank 和跨实例的事务（D9）、DP/EP 弹性；平台插件接入第二种加速器（如昇腾）。

### 13.2 验证指标

- 每 GPU 小时产出多少个经过验证的改进。
- 每轮迭代的延迟分布：重启、评测、agent 思考各占多少。
- 误接受率：通过门禁、但之后发现性能回退或正确性问题的改动比例。
- **迁移率**：新内核上找到的改进，有多少能在成熟开源引擎（vLLM/SGLang）上复现。新内核早期的基线较弱，改进本来就更容易找（SEIS 的 3.27 倍也是相对弱基线），所以这一项最关键。
- 包络失效的次数和类型。
- 可选的对照组：
  - 在 vLLM 或 SGLang 上只加一层薄的 RSI 层（4.1 中不作为主路线的方案），用同一个 agent、同样的预算跑一遍，量化新内核到底带来了多少增益。
  - 在 mini-sglang 上用同一个 agent、同样的预算跑一遍 SEIS 式的无约束进化，对比误接受率，量化契约和参照模式带来的收益。

---

## 14. 待决问题

1. 第二阶段的第一个模型：选 Qwen3 dense，实现更快，而且 nano-vllm、mini-sglang 都有现成的对照；还是 Qwen3.5 混合架构，它是当前主力代际，从第一天就能压测 D4。
2. 机制层用 Rust 还是 C++（D2）。
3. 公开 trace（Mooncake、Azure、BurstGPT 等）是否足够做 e2e 工作负载？是否需要定义一个 trace 格式，让部署方贡献脱敏的请求级 trace？
4. policy 层先做哪一类：投机解码、graph capture，还是调度参数。DeepInfer 的 DSpark 和 prefill 预算求解是现成的强基线，值得优先对标。
5. 和 vLLM、SGLang、DeepInfer 社区的协作方式：评测平面是否单独发布、是否贡献给上游。
6. 评测平面的模型级 SOL 下界能否做到足够准。
7. 许可证（选 Apache-2.0 可以和 vLLM、DeepInfer 保持一致，便于移植代码）和治理模式。
8. 参照模式怎么处理近似重建类的优化（如 SWA bounded replay）：关掉，还是把缓存状态纳入评测输入？

第 1、2 条已在第 15 节决定：第一批模型是 DS 1017 和 Qwen3.8-27B；v0 全部用 Python，预留 FFI 边界。第 4 条不再由框架决定，见第 15 节第 1 条。

---

## 15. 2026-11-04 讨论：v0 计划

完整的决策表、架构、路径、里程碑见 [v0-plan.md](v0-plan.md)。和前文相比，主要的变化有：

1. **框架不预设 RSI。** 优化目标、agent、LLM、可改范围都由使用者决定。框架提供机制：安全地应用和回滚任意改动、身份和谱系、评测隔离、冷启动复现。原来的策略决策点、kernel 注册表、两档门禁，都降为默认插件和加速路径。
2. **引擎本体是产品。** 原版和经过 RSI 改进的版本都必须像 vLLM 一样可以部署。RSI 平面可以拆卸，拆掉之后引擎就是一个普通的推理引擎。线上不重启的调整能力属于内核本身。
3. **Vite 式开发体验。** agent 启动引擎，修改工作区，用显式的 `apply` 应用改动。框架自动判定档位并执行，按版本切换，每个响应都带系统身份。"是否真的更好"的结论只来自冷启动的独立评测实例。
4. **起点仍是新内核**（维持 4.2）。Qwen3.8 参考 HF 和 vLLM 的实现移植；DS 1017 的模型代码和 kernel 从 DeepInfer 移植，放在 v0 的 M5。
5. **v0 直接接真实后端，不写模拟器。** 原先"语义哈希执行器 + 代价模型"的方案已经撤回，原因是工作量可能比接真实后端还大，而且是一套和真实实现分叉的第二实现。v0 在单台 8×SM100 上先跑通 Qwen3.8-27B，再跑通 DS 1017。快速测试用同一份模型代码，配上 PyTorch 参照 solution 和小尺寸配置，只要求能在 SM100 上跑；CI 只用一个极简的假执行器测变更机制，不含任何模型语义。
6. **D4 的包络失效日志第一条**：DS 1017 用了跨层 KV 共享（`kv_source_layer_ids`），而 11(1) 原本把它列为包络外。状态归属在 v0 就要支持"借用另一层的状态"。
7. **v0 不做鉴权和角色，也不区分线上和开发。** 所有实例都通过同一个 `apply` 入口接受任意改动，调用方可以指定档位上限。评测器只保留结构上的分离：代码不在候选快照里，评测在独立进程中运行。D12 里的权限、按改动类型的白名单、发布签名，都推迟到后续版本，通过变更流程里预留的策略钩子接入。
8. **评测采用类似 SWE-bench 的 patch 提交。** agent 在常驻实例上用 `apply` 自由开发；要评测时，向评测服务提交（起点身份, patch, 套件）。patch 是对身份清单的 diff。评测服务负责校验（包括套件声明的受保护路径）、物化、冷启动和成对测量，结果按（起点, patch, 套件版本）写入谱系。基线前进后，patch 必须先 rebase 再重新评测。`apply` 和评测共用同一种 change-set 格式。这取代了 10.5 和 10.7 里"温路径加快照"的评测思路：评测一律冷启动，热、温路径只服务于开发时的常驻实例。
9. **性能不是 v0 的目标。** v0 可以很慢，但架构必须正确。我们不知道使用者的 RSI 需求，预先做的优化可能反而妨碍 RSI：使用者的 agent 要先把它们改回去，再做自己的优化。判断一个特性要不要进 v0，看它会不会对状态、plan 或变更机制提出新的要求：对架构有要求的特性只做最简单的正确版本；只影响快慢的优化（异步调度、kernel 融合、调度策略调优等）留给使用者。**CUDA graph 是例外：必须实现，decode 默认开启。** 因为它对架构的约束最强（静态 buffer、冻结的绑定、和双缓冲 plan、跨进程显存、版本切换的交互），而且真实部署都会用。这也修正了 13.2 的说法：迁移率仍然值得度量，但它由使用者决定是否关心，不再是框架本身的验收指标。v0 还包含最简版本的前缀复用和固定长度投机解码；prefill 先用 eager。
10. **v0 不开源。** 要等优化过、反复测试过、足够成熟之后再开源，到那时再和 DeepInfer、vLLM 做性能对比。这一条取代第 0 节"作为开源项目发布"和 2.1"从第一天起开源"的说法，但 D14 的约束保留：内核里不允许出现私有依赖，保证以后能开源。
11. **其他细节**：
    - change-set 按文件内容整体替换；
    - 性能默认成对测量，套件可以改成复用缓存的基线；
    - **资源分配不归引擎管**：常驻实例由 agent 用自己拥有的 GPU 自行启动、终止、更新；评测服务由人在自己选定的机器上启动，只使用启动时指定的 GPU。

    这取代了 10.7 里"把区间独占计时扩展到多卡和多节点"的资源调度设想。
12. **版本切换、环境变量、多模态、鉴权**：
    - 同一时刻只有一个版本在执行，不做新旧版本交替执行（例如旧版本 TP4、新版本 TP2 时，交替执行要同时维护两套权重布局和 KV 布局，太复杂）。新版本后台准备好以后排空旧版本；排空超时可配置，默认 10 秒，超时后中止剩余的在途请求，再切换。
    - v0 不限制读取环境变量。环境变量不计入身份；评测服务用一组固定的环境变量冷启动，并记录下来。这取代了 3.9 第 9 条和 D12 里"禁止裸 getenv、环境变量全部进身份"的要求。
    - v0 只做文本生成，DS 1017 的视觉编码器推迟到 v1。
    - 鉴权、角色、线上和开发的区分、发布签名，放到 v1、v2 或更晚，v0 只保留策略钩子。
13. **不分热、温、冷三档，按实际影响推导动作。** 和 Cordis 一样，从"改了什么"推导出"哪些东西要替换、重建或重启"，只动真正受影响的部分：改 TP 只重建 rank 进程；同一模型换权重只做原地 refit、让 KV 失效；扩大 KV 池只扩容，不重启也不排空。排空也改为按需：只有动作会影响在途请求时，才排空受影响范围内的请求。调用方只有 `apply --dry-run`，用来先看动作计划。这取代了 6.2 的三档划分和 D3 里"档位自动判定"的说法。

---

## 附录 A：关键数字速查

| 项目 | 数字 | 来源 |
|---|---|---|
| vLLM、SGLang 最近 30 天合入的 PR 数（2026-10-04 至 11-03） | 1,928 / 2,283 | GitHub API |
| Python 代码体积（含测试） | vLLM 约 78 MB、SGLang 约 106 MB、mini-sglang 约 0.29 MB（约 5 千行）、nano-vllm 约 0.05 MB | GitHub API、LMSYS 博客 |
| SGLang 热修改参数和重启的耗时对比 | 16–22 毫秒，对比 56–86 秒 warm、141–391 秒 cold | SGLang PR #37661 |
| vLLM Elastic EP 的停顿时间 | 初版 56–100 秒，优化后 0.82–1.96 秒（DeepSeek V4 Pro） | vLLM 博客 |
| flashinfer-bench `apply()` 的开销 | 每次调用 1–2 微秒，端到端低于 0.8% | FlashInfer-Bench 论文 |
| DynaTrain 并行度在线切换 | 70B 不到 2 秒，235B MoE 4.36 秒 | arXiv 2605.18815 |
| Pollux 用 checkpoint-restart 重新分配资源的耗时 | 15–120 秒 | OSDI'21 |
| Atrex-Bench、KREX、Herschel | 见 1.2 节 | 公开论文 |
| DeepInfer 代码量 | `deepinfer/` 约 4.6 万行 Python，`csrc/` 约 5.9 千行 C++，测试约 3.1 千行 | 源码 `wc -l`（commit a679bb8） |
| DeepInfer 公开仓库状态（截至 11-03） | 约 25.1k star；6 个 commit；合入 2 个外部 PR；约 210 个 open PR | GitHub API |
| DeepInfer 的配置面 | 集中定义 59 个环境变量，另有少数直接 `getenv` | 源码 |
| DSpark 变长验证 | 验证 token 数减少约 30%，平均接受长度 3.29 → 3.20；每 rank 60 并发时吞吐 +5.4%（16 并发 −1.5%，96 并发 +11.0%） | DeepInfer Q&A |
| DeepInfer 的 EPLB | prefill `moe_lb` 1.045–1.074，decode 1.090 | DeepInfer Q&A |
| DeepSeek V4.1 的 KV 存储 | 3FS 约 100 PiB SSD；TTL agent 3 天、chatbot 7 天；兼容键 `(部署配置, 权重, 用户组)`；未做过格式迁移 | DeepInfer Q&A |
| nano-vllm / mini-sglang | 约 16.3k / 5.5k star；`nanovllm/` 1,385 行 / 去空行注释约 6 千行；2026-08 以来都没有提交 | GitHub API、源码 |
| SEIS | Qwen3-0.6B、H100、单请求：594 → 1,942 tok/s（3.27 倍），关掉投机 3.08 倍；数值门禁 ppl ±5%、KL ≤ 0.05、argmax ≥ 90%；一个通过门禁的引擎省略了约 39% 的词表行 | arXiv 2610.04646 |

## 附录 B：动态参数和并行度相关研究（按主题列出）

- **serving 并行度**：SpotServe（ASPLOS'24）、LoongServe（SOSP'24，elastic sequence parallelism）、Seesaw（MLSys'25，相对 vLLM 平均 1.36 倍）、DynamoLLM（HPCA'25）、Gyges（2509.19729，多个 TP1 合并成 TP4 再拆回）。
- **chunk 大小和 token 预算**：Sarathi-Serve（OSDI'24，静态预算）、SLOs-Serve（2504.08784）、DynaServe（2504.09285）、AdaptoServe（HPCC'25）、P-PAS（2608.15171：不存在单一的静态最优 MBT）；DeepInfer 的时间预算加二次代价模型是一个生产实例。
- **配置搜索**：SCOOT（WWW'25）、Vidur（MLSys'24，模拟器）、agent 化的 autooptimizer 类工具（每轮都重启 server）。
- **RL 中的动态**：PAT（ICPP'26，在 generation 长尾阶段自适应 TP）、RollPacker、Earl、OrchestrRL；rollout 预算分配方面有 Knapsack RL（ICML'26）、HORA、CERO、DAPO 的 dynamic sampling；DeepScaleR 的 8K→16K→24K 实际上是三次独立的训练。
- **训练**：PBT、KungFu（OSDI'20）、Pollux（OSDI'21）、HotSPa（SOSP'24）、Tenplex（SOSP'24）、DynaTrain、Megatron Dynamic-CP。

## 附录 C：参考链接

- Atrex-Bench：https://arxiv.org/abs/2607.14541 ，https://github.com/alibaba/atrex-bench
- AKA：https://github.com/alibaba/atrex-kernel-agent
- KREX：https://arxiv.org/abs/2609.30057
- Herschel：https://arxiv.org/abs/2609.40247
- FlashInfer-Bench：https://arxiv.org/abs/2601.00227 ；FlashInfer Trace Apply：https://docs.flashinfer.ai/trace_apply.html
- Cordis 论文：https://arxiv.org/abs/2608.25512 ；DeepSeek Harness：https://github.com/deepseek-ai/deepseek-harness
- DeepInfer：https://github.com/deepseek-ai/DeepInfer ；V4.1 Flash 部署 Q&A：https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md ；deepseek-recipe：https://github.com/deepseek-ai/deepseek-recipe
- DeepInfer 性能对比：vLLM（https://vllm.ai/blog/2026-10-23-deepseek-v41-flash-pd-wide-ep ）；SGLang（https://www.lmsys.org/blog/2026-10-24-dsv41-deepinfer/ ）；GPUStack 8×SM100（https://gpustack.ai/blog/ ）；8×B300 单机对比（https://dev.to/kbespalov/deepseek-v41-flash-on-8xb300-vllm-vs-deepinfer-600k-tokenss-3jl9 ）；对 DeepInfer benchmark 口径的分析（https://theservingstack.com/matching-deepinfer ）
- DeepSeek 关于开源推理引擎的说明（2025-04）：https://github.com/deepseek-ai/open-infra-index/tree/main/OpenSourcing_DeepSeek_Inference_Engine
- nano-vllm：https://github.com/GeeeekExplorer/nano-vllm
- mini-sglang：https://github.com/sgl-project/mini-sglang ；LMSYS 博客：https://www.lmsys.org/blog/2025-12-17-minisgl/
- SEIS：https://arxiv.org/abs/2610.04646 ；项目页：https://nehzux.github.io/SEIS/
- vLLM Elastic EP：https://vllm.ai/blog/2026-05-14-elastic-expert-parallelism ，https://vllm.ai/blog/2026-10-19-elastic-ep-near-zero-downtime
- vLLM PR #30316、#61287，RFC #51608；SGLang PR #37661
- ADRS（Barbarians at the Gate）：https://arxiv.org/abs/2510.06189
- ISO-Bench：https://arxiv.org/abs/2602.19594
- DynaTrain：https://arxiv.org/abs/2605.18815
- DeepSeek-V4：https://arxiv.org/abs/2606.19348 ；Qwen3.5（transformers 文档）：https://huggingface.co/docs/transformers/model_doc/qwen3_5
