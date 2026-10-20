# 0.3.0

拆分独立 cage4-baselines、只读 Studio Server 和 macOS SSH Client。提供 Random/Sleep 与真实五策略 Actor-Critic，独立 checkpoint 评估和 seed 统计；算法注册不冒充 IPPO/MAPPO 实现。

新增只依赖 agent 实际 observation 的编码、逐步动作/观测与红队 Session 记录、真实接口链路拓扑、分项损失曲线、跨 Episode/Step 回放和实验对比。新增 cage4.run/1 原子记录、准确的进程身份检测、macOS 生命周期管理及回滚。

保留既有安装与实验，归档 v0.1.4 源码，提供中文安装/研究/协议/验证文档和 macOS CI。未改变仓库可见性或替用户添加许可证。
