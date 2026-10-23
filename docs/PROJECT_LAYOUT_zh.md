# 项目、部署与旧文件

当前唯一研究入口：服务器 `/Users/tiankedi/cage4-research`，GitHub `tkdzombie/cage4-studio`。项目目录名与仓库名不同，环境都是CAGE4。原实验的source快照不覆盖，新的入口保存将来的算法版本。

不能删除：`.venv`及其目标`.venv-independent`、独立基础环境`/Users/tiankedi/cage4-runtime-independent`、`cage-challenge-4`、`Hierarchical-MARL`（含初始化权重）、`runs`、`deliverables`、当前源码与实验评估JSON。依赖两个上游的锁定commit见 environment-independent.json；tracked源码维持原样。

旧CAGE3项目和runtime现已不被CAGE4导入，其8502只读看板已停用，LaunchAgent已从自动启动目录移入maintenance-backups/retired-cage3-dashboard。清理以可恢复归档为主；不批量删除历史模型/日志。`.venv-shared-legacy`是旧CAGE4解释器回滚目录，日常不用。新的训练不启动旧队列、旧自动服务或CAGE3。

本仓库旧平台源码移到 `legacy/previous-platform/`，旧文档里的启动命令只代表历史。`work/cage4-hmarl`是本地已验证结果的备份，本地日常修改用 `work/cage4-studio`，然后同步服务器。不要同时在两份源码各改一套。

服务器本次更新前源码快照在 `maintenance-backups/`（不含重复模型/venv），保存恢复所需旧入口。本机CAGE3客户端已移入 ~/Applications/Archived Research Clients/；当前使用 ~/Applications/CAGE4 Studio Client.app。

## 迁移到新机器

当前服务器环境已经可用，无需重装。新机器需先准备Python3.10，并在项目根目录拉取以下上游和锁定commit：

```bash
git clone https://github.com/cage-challenge/cage-challenge-4.git
git -C cage-challenge-4 checkout 8c3c50ca54b176c2de199847944e8dcc035497e3
git clone https://github.com/adityavs14/Hierarchical-MARL.git
git -C Hierarchical-MARL checkout 6fe960f5931f2d7dc5ef522f09ca4a179f377ee4
```

`runtime-freeze.txt`是实际服务器包快照，其中的conda构建路径不是可直接移植的requirements。`requirements.txt`列核心版本，安装CybORG后若其上游依赖覆盖gym/Ray等需再核对核心版本；新机器安装尚未作为本次验收目标。以现有服务器解释器为准。用 `.venv/bin/python verify_flat.py`完成499步环境核对，再用小预算运行训练和模型恢复，不将依赖安装成功等价为学习验证。

服务器根目录的64份旧诊断脚本/JSON/日志已移入 `archive/diagnostics-20261023/`，当前两份终点评估、配置和源码保留在主目录。服务器现有目录已关联同一GitHub仓库和main分支；venv、上游源码、checkpoint和归档均不提交Git。
