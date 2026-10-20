# 迁移审查（2026-10-20）

实查对象：用户确认的 tkdzombie/cage4-studio 仓库（main fddedb908e6d102c9df70960c6100dda3a7ed16b），以及 Mac Studio $HOME/cage4-studio 的 v0.2.0 安装。后者不是 Git checkout，包含 .conda-env、cage-challenge-4、runs、重复解压目录和 __MACOSX。本地 /mnt/data 不存在；在前一日 Codex 输出目录找到 v0.1.4 ZIP，但没有拿它覆盖服务器。

现有训练代码确有 optimizer.step，属于独立五策略的 episodic REINFORCE + value baseline，不能称为 PPO/IPPO/MAPPO。原训练模型定义在 execute 内，缺少独立 evaluate、完整分项损失和 seed 统计；网页可直接创建训练进程，研究代码与观测台耦合。

原 BlueFlatWrapper 内部读取 state.hosts 的事件和任务阶段。0.3.0 改用 BlueFixedActionWrapper 提供官方动作索引和掩码，仅将 get_observation(blue_agent_i) 的实际返回值编码为策略特征。固定动作 wrapper 内部仍利用官方环境元数据构造动作空间：允许动作掩码属于公开给 Blue 的接口信息，不是自定义全局状态特征。策略调用只接收 PolicyInput，不接收 adapter/base/遥测记录。此边界是代码与数据流隔离，并不是对恶意研究插件的进程级沙箱。

迁移方案：旧安装和实验原地保留；新源码进入独立版本目录。复用已验证 Python，不重装大依赖，不修改原 .pth。launchd 只切换 dashboard 的程序路径；原 plist 备份并可还原。既有 TensorBoard 不改，研究进程不被服务脚本停止。旧数据只读显示，旧版 running PID 因没有进程出生身份显示 legacy_unverified。

GitHub 历史中的 v0.1.4 文件归档在 legacy/v0.1.4；服务器 v0.2.0 的原训练参考保存在 legacy/v020_train.py（不可作为新入口执行）。原仓库没有 LICENSE，未擅自添加授权。CybORG 仍作为外部依赖，遵守其自己的许可。
