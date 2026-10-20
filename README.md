# CAGE4 Studio 0.3.0 · 独立科研工程

研究项目决定算法，Server 读取实验，Client 连接网页。面向 macOS / Apple Silicon / Toolbox 远程 PyCharm；非官方 CAGE 工具，不声称训练收敛或达到挑战基准。

| 部分 | 目录 | 职责 |
|---|---|---|
| cage4-baselines | `cage4-baselines/` | 环境适配、YAML、train/evaluate、Random/Sleep、五策略 Actor-Critic、插件注册 |
| Studio Server | `server/` | 只读曲线、五 Agent 指标、真实接口拓扑、红队记录、Episode/Step 回放、对比 |
| macOS Client | `client/` | 本地 SSH 转发与浏览器；不启动 PyCharm或服务器 |
| 共享协议 | `protocol/` | cage4.run/1 原子文件格式、进程身份与状态 |
| 运维 | `ops/` | 服务管理、研究入口、验收与打包 |

已实查服务器旧环境和用户原 GitHub 仓库。迁移不覆盖 `$HOME/cage4-studio`；原数据、环境、CybORG 源码都保留。旧项目源码在 Git 历史与 `legacy/` 留存。没有替用户选定 LICENSE，见 [NOTICE](NOTICE.md)。

## 现有 Mac Studio 快速使用（远程终端）

以下使用当前远程用户的 $HOME 路径；SSH 主机与用户须按自己的配置填写。公开源码不包含个人连接信息。

```bash
cd $HOME/cage4-studio-releases/CAGE4_Studio_v0.3.0
bash ops/check.sh
bash ops/research.sh train \
  --config cage4-baselines/configs/actor_critic.yaml \
  --runs $HOME/cage4-experiments
```

从命令输出找到本次 run 目录，再使用其 checkpoint。示例中的 `<RUN>` 必须换成真实目录名：

```bash
bash ops/research.sh evaluate \
  --config cage4-baselines/configs/actor_critic.yaml \
  --checkpoint $HOME/cage4-experiments/<RUN>/checkpoints/episode_000030.pt \
  --seeds 1001 1002 1003 \
  --runs $HOME/cage4-experiments
```

评估的 episodes 也来自 YAML；可以复制配置并将 episodes 改成 5。Random/Sleep 用对应 YAML，evaluate 时不传 checkpoint，使用相同 seeds、步长和红队进行对照。训练每个 seed 独立运行、独立目录；评估不更新参数。Ctrl+C 在步间安全停止，网页刷新或关闭不会停止研究进程。

## 本地 MacBook Air

双击下载包中的 `CAGE4 Studio Client.app`，建立本地 18501 → Mac Studio 8501 隧道并打开网页。应用无 Python 运行依赖；需现有 SSH 密钥登录、已确认主机指纹、服务器服务已启动。个人应用未签名/公证，按 macOS 的“右键 → 打开”流程允许使用。首次 SSH 配置只能在确认服务器指纹后执行，客户端不会绕过主机校验。

客户端退出后 SSH 隧道继续运行，重开复用同一 ControlMaster。停止隧道用本地终端，在客户端 app 所在目录执行：

```bash
bash "CAGE4 Studio Client.app/Contents/Resources/connect.sh" stop
```

需要 TensorBoard 转发时用 `bash client/connect.sh start studio.example researcher 18502 16006`，网页从本地 18502 打开，TensorBoard 从本地 16006 打开。默认不转发 TensorBoard。不同参数建立不同隧道；对应 stop 必须使用同一组参数。客户端不自动打开 PyCharm，也不要求安装第二套 PyCharm。

## Toolbox 远程 PyCharm

本地 Toolbox → Remote Development → SSH `researcher@studio.example`，打开远程版本目录下的 `cage4-baselines`（不是网页、不是本地解压目录）。本次提供独立轻量研究解释器 `$HOME/cage4-research-env/bin/python`，它复用旧环境的大依赖，研究代码不在 Server 中运行。

在 PyCharm 标记 `src` 为 Sources Root；`protocol` 是相邻的共享包。独立环境已安装这两个本地 editable 包。共享 `.run/Train Actor Critic.run.xml` 作为运行配置参考；若 JetBrains 版本没有自动识别，手动创建 Python module 配置：模块 `cage4_baselines.runner`，参数 `train --config configs/actor_critic.yaml --runs $HOME/cage4-experiments`，工作目录为项目根目录。无需修改 Client。

## 安装、升级、回滚、扩展

- [本地/远程分步安装和回滚](docs/INSTALL_zh.md)
- [日常研究与添加算法](docs/RESEARCH_zh.md)
- [版本化文件协议与安全边界](docs/PROTOCOL.md)
- [真实代码审查与迁移依据](docs/AUDIT.md)
- [验证结果与兼容性限制](docs/VALIDATION.md)

M3 Ultra 不等于 CUDA GPU。默认 CPU；MPS 只加速支持的 PyTorch 算子，不加速整个仿真。IPPO/MAPPO 尚未实现，注册表对这些名字明确报错。Actor-Critic 是 Monte Carlo REINFORCE + value baseline，不是 PPO。当前不支持断点续训；旧 BlueFlat checkpoint 不兼容新 observation encoder。完整训练、测试、可运行包与部署状态以验证文档为准。
