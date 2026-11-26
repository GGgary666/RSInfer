# 重启时节点代理撞到上次的进程目录

- 日期 2026-11-26
- 状态 已修，Ascend 双 worker 同路径重启通过
- 引擎 `dev/ascend_a3_gg` 上的 GLM-5.3-Flash，四层诊断，Ascend 双 worker。初次启动身份 `7c1a8441950c54c7`
- 操作 用 `stop --run` 停止服务，再以同一个 `--run` 执行 `serve`
- 现象 新节点代理创建 `.run/frontend-0-0` 时抛出 `FileExistsError`，启动事务失败
- 原因 进程目录编号从节点代理启动时的零开始，关停会保留日志目录
- 修复 每次拉起进程时用 `mkdtemp` 原子新建带角色和 rank 前缀的目录，原日志继续保留
- 绕过 使用新的 `--run` 路径启动

同路径复测中，原引擎 PID 为 2271865，新引擎 PID 为 2279293。两次的 frontend、scheduler 和各 worker 目录均保留。新 worker 从原 MemCache 池恢复状态，重放的 701-token 请求命中 640 token，生成文本一致。
