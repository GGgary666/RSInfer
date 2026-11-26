# GLM 旧前缀清单在热更时被再次导入

完整 TP8 MTP3 使用 `--prefix-manifest` 跨进程恢复后，批量 KDA 候选通过 `check` 与 `apply --dry-run`。`apply --wait-empty` 在 8 个 worker 的模型重载阶段都收到 MemCache `-3007`，事务回滚至 `b66052d0435a00b9`。 [计划和返回值](../../../ascend-a3-results/2026-11-26/glm-native-mtp3-tp8/glm-native-aisbench-mtp3-packed)

缓存单元直接依赖层服务。层代码热更使缓存单元随之重建，丢失已导入标记。新实例用候选代码的指纹读取旧启动清单，外部池中只有旧指纹的对象。即使指纹一样，再导入启动时的页号也可能覆盖在用的新状态。

把层指纹由模型启动传给缓存服务，移除缓存服务对层实现的依赖。启动清单只在 worker 和调度器各自的初始单元代加载。热更按既有排空事务刷新本地 radix，后续检查点使用当前模型及层指纹。

用修复后的 45 层 TP8 MTP3 重新保存前缀，在原池上重启。8 个 worker 都换了 PID，首条回放命中 2,048/2,073 token，输出与重启前一致，换盐为 0 命中。再次 `check`、`apply --dry-run`、`apply --wait-empty` 成功，identity `0dcb8cffa5cc63bd` → `ceabc12dc403f177`。暂停 0.021 秒、execute 2.715 秒，8 个 worker PID 保持相同。 [重启和换盐证据](../../../ascend-a3-results/2026-11-26/glm-native-mtp3-cache-hot-recheck/cache-proof/restored.json) · [热更事务](../../../ascend-a3-results/2026-11-26/glm-native-mtp3-cache-hot-recheck/glm-native-aisbench-mtp3-packed-fixed)

重启新增成功 GET 136 次。热更、完整 AISBench 和撤去候选后，池的 GET 计数保持在 3,529。撤去候选没有更换 worker，回滚后新请求正确输出该前缀中的验证码。 [全过程证据](../../../ascend-a3-results/2026-11-26/glm-native-mtp3-cache-hot-recheck/cache-proof)
