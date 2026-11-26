# Ascend worker 找不到 CANN 的 TBE 模块

- 日期 2026-11-26
- 状态 已修，真实 DFC 权重启动和四条请求验证通过
- 引擎 GLM-5.3-Flash 四层 TP2，CANN 9.2.0，W8A8，`--glm-moe-backend dfc`
- 现象 构造 FRACTAL_NZ 专家权重时，CANN 初始化 GE 报 `ModuleNotFoundError: No module named 'tbe'`。两个 worker 装载模型单元均失败
- 原因 进程宿主只保留快照作为 `PYTHONPATH`，丢掉 CANN `set_env.sh` 的工具链 Python 目录。小尺寸拷贝使用了另一条算子路径
- 修复 节点代理在 Ascend worker 的快照路径后加入已配置 CANN 安装的两个 Python 目录。权重加载显式在设备转换到 NZ，随后进行同格式复制

原运行目录复测通过。前四层使用真实 int8 专家，15、85、335、705 token 的请求均完成，与 grouped GEMM 路径的贪心文本一致。
