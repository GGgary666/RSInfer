---
name: rsinfer-ascend-kernel
description: 为 RSInfer 选择或接入 torch_npu、ACLNN 与 Ascend C 算子，调整 tiling、布局或融合，在目标形状与真实调用点验证派发、精度和收益。
---

# RSInfer Ascend 算子接入

## 从真实调用点定契约

先找到实际 worker、当前实现、调用频率、目标 prefill/decode 形状和参照。Qwen3 的分页 attention、KV 写入、RMSNorm 等由 `kernels.ascend` 通过 `kernels.catalog@1` 注册方案。GLM 的 KDA、稀疏 MLA/DSA 与 MoE 从 `layers.glm53`、`models.glm53` 和加载的扩展追。 [GLM 层](../../../engine/layers/glm53/unit.py) · [kernel 目录](../../../engine/core/kernels/unit.py)

记录实际 shape/stride、dtype、ND/NZ 或分页布局、量化 scale/offset、mask/block table/slot mapping、有效长度和 alias / 原地写入。多 rank 另记 shard、通信组、count 与输出归并。用本机 CANN 和目标 SoC 支持的接口、布局及取值范围核对候选。

## 选择最小的可验证实现

- 可复用已有 `torch_npu` 或 ACLNN/扩展算子时，将重排、cast、拷贝和同步算入调用点收益。检查 alias、状态和数值语义，尤其是缩放/累加 dtype、tie-breaking 和 padding。
- 需要 Ascend C 自定义实现时，按目标 SoC 与集成接口选择直调验证或正式注册接入。根据 shape 和运行时约束设计多核分片、UB/L1 tiling、对齐、尾块、buffer 和同步。按该 SoC 核对片上容量和指令支持，用边界用例验证多核写入，用 profile 检查 Cube/Vector/MTE 重叠。 [CANNBot tiling 设计](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/ops/ascendc-tiling-design/SKILL.md)
- 以 Qwen3 为例，`attn.block_paged@1` 与 `kv.store@1` 的 Ascend 实现要求 `fp32=0`、`kv_as_q=1`、页长 128，head dim 为 64、128 或 256。用匹配的 BF16 配置和支持的 FP32 对照，在 worker 的 `core.plan` 日志中按 prefill/decode 核对最终 solution ID，再用请求确认命中。catalog 的 `auto` 可能选择 reference，显式 backend 无匹配时 `select` 会抛 `LookupError`。 [注册条件](../../../engine/kernels/ascend/unit.py) · [attention 后端选择](../../../engine/layers/attn/unit.py)
- 沿用目标层已有的派发边界。对录图路径验证固定地址、可捕获输入及热更后是否需要重录。

## 三层验收

1. 对真实形状做算子级正确性和重复计时，同时覆盖边界及邻近形状。用相同输入、预热和同步边界。需要上板指标时按本机工具查 `msprof` 的支持情况，多 rank 用完整应用采集。 [CANNBot 算子性能采集](https://gitcode.com/cann/cannbot-skills/blob/63f4f18a33dfb116125360ff8269c037b69f9958/ops/ops-profiling/SKILL.md)
2. 在实际调用点核对布局转换与额外分配、launch、同步的成本。对原地更新，在克隆的状态上跑参考并比较修改范围、输出及后续步。用热更验证候选的真实派发与回退。 [精度与运行故障](../rsinfer-ascend-debug/SKILL.md)
3. 以相同负载复核服务正确性、吞吐和延迟，按有收益的阶段或 shape 放行。已有 AKA task ID、候选或确实需要做算子搜索时，沿用 AKA 契约和回接流程。 [服务性能验收](../rsinfer-ascend-optimize/SKILL.md#验收并记录) · [AKA](../optimize-kernels-with-aka/SKILL.md)
