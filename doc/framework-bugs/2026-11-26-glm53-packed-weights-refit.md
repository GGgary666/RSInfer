# GLM 原地 refit 后，已录制的图继续读取旧的 MLA 合并权重

- 日期　2026-11-26
- 报告人　Codex
- 状态　已修，CPU 回归通过
- 引擎　原生 GLM-5.3-Flash，启用 `glm53.mla_fused_proj` 和 decode ACL Graph。
- 变更　执行 `core.state.weights` 的 `refit`，将检查点载入现有权重张量。
- 现象　MLA 的两组合并矩阵由 `torch.cat` 派生，拥有独立存储。原地加载更新了源张量，已经录制的图仍然引用合并前的旧内容。
- 期望　新前向读取 refit 内容，合并矩阵的地址保持稳定。撤回 refit 时，由同一个加载器恢复旧内容。
- 修复　权重加载成功后原地刷新已分配的合并矩阵。录图和回放继续用同一地址。
- 验证　CPU 回归覆盖加载新权重、回到旧权重和地址不变。 [回归测试](../../tests/units/models/test_glm53_graph_metadata.py)
