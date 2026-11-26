# HCCL 独立分组销毁后重建卡住

四个真实 910C die 分成两组，各组在第一次创建时成功做 BF16 all-reduce、变长 all-gather 和 Gloo 规约。使用 `use_local_synchronization=True` 销毁并重建同一组成员后，第二轮的 HCCL 建组没有结束。CPU 的 Gloo 测试可完成两轮。 [首次分组日志](../../../ascend-a3-results/2026-11-26/validation/comm-domains-hccl.log)

`worker.comm` 让所有 rank 以固定次序创建每个子组。PyTorch 为每次建组分配新的 rendezvous 序号，各进程只持有自己参加的子组。实机更换端口范围后，两组连续创建、核对并销毁两次。每组只规约自己的两个 rank，分别得到和 1、5。 [修复后复测](../../../ascend-a3-results/2026-11-26/validation/comm-domains-hccl-recheck.log)
