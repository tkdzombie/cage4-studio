# 基础环境与可运行 Baseline 入口

修正环境 reset/step 状态管理，严格校验每个 Blue 动作与回合生命周期。新增 smoke、quickstart 和可执行环境示例，一条命令实际运行 Actor-Critic 训练、权重验证及 Random/Sleep 多 seed 对照。增加真实环境契约测试与中文跑通指南。

# 后续研究工程整理

按 core/envs/algorithms/experiments/recording 拆分研究代码；Python 依赖说明、训练/评估/基线 YAML、项目测试与 PyCharm 配置各归其位。新增完整系统使用说明书和目录职责文档，保持 v0.3.0 导入、CLI、配置及 checkpoint 兼容。固定部署和 run schema 不变。

# 0.3.0

拆分独立 cage4-baselines、只读 Studio Server 和 macOS SSH Client。提供 Random/Sleep 与真实五策略 Actor-Critic，独立 checkpoint 评估和 seed 统计；算法注册不冒充 IPPO/MAPPO 实现。

新增只依赖 agent 实际 observation 的编码、逐步动作/观测与红队 Session 记录、真实接口链路拓扑、分项损失曲线、跨 Episode/Step 回放和实验对比。新增 cage4.run/1 原子记录、准确的进程身份检测、macOS 生命周期管理及回滚。

保留既有安装与实验，归档 v0.1.4 源码，提供中文安装/研究/协议/验证文档和 macOS CI。未改变仓库可见性或替用户添加许可证。

## 0.4.0

- 接入锁定作者源码的 IPPO、MAPPO、HAPPO、MAPPO-Lagrangian、MACPO 统一训练/评估与checkpoint。
- MACPO 离散 Categorical KL / 数值 CG 适配、顺序 importance factor、真实梯度与完整任务阶段验收。
- 阶段业务加权 cost、独立40场景校准/验证、三档预算、单位/种子/信息权限协议。
- 官方团队 reward 计一次与旧五Agent之和分开，网页增加业务成本/预算和可比较的回报选择。
- 新增中文论文算法与cost预算说明，保留基础环境/Actor-Critic入口和旧实验兼容。
