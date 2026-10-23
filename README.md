# CAGE4 Studio · 算法实验起点

当前使用 **CAGE4**。环境、完整回合、PPO学习、模型保存/恢复与共同场景评估已跑通，可以直接开始写自己的算法。

服务器项目：`/Users/tiankedi/cage4-research`。PyCharm 打开这个目录，解释器用 `.venv/bin/python`（独立CAGE4环境，不依赖CAGE3）。

```text
algorithms/my_algorithm.py      ← 写自己的 loss / PPO 扩展
experiments/run_baseline.py     ← 选算法、初始化、运行名，右键 Run
train.py                       ← 公共训练、采样、保存入口
primitive_env.py                ← 固定的 CAGE4 环境适配
compare_flat.py                 ← 模型恢复与共同场景评估
runs/<运行名>/                  ← 新实验配置、源码、指标、模型（不提交Git）
dashboard.py                   ← 只读看板
```

先读 [算法修改指南](docs/ALGORITHM_GUIDE_zh.md)。模板原样等价于PPO，不冒充新算法。默认初始化 author；从零研究时改 scratch，两个比较方法必须使用相同初始化和预算。

现有主要基线 `runs/flat-full-actions-author-init-s321/final`，单训练seed321，30更新/359280交互步，四开发场景回报从 −2035.25 到 −370.5。另保留冷启动PPO。详见 [结果](RESULTS.md)、[修复与失败记录](AUDIT.md)。这是实用baseline，不宣称论文性能复现。

[环境协议和已执行命令](docs/BASELINE_PROTOCOL_zh.md) · [目录与旧文件说明](docs/PROJECT_LAYOUT_zh.md) · [运行环境指纹](environment-independent.json)

## 看板与 app

CAGE4 服务为服务器 loopback 8501。Mac 客户端只建 SSH 隧道并打开浏览器，不训练。新客户端名 **CAGE4 Studio Client.app**；CAGE3 客户端指向旧8502，请勿混用。

```bash
# 本地 Mac：按实际 SSH 主机和用户构建；产物留在 client/build（Git忽略）
bash client/build_mac_app.sh client/build YOUR_SSH_HOST YOUR_USER 18501
```

服务器手动启动看板：`bash ops/dashboard.sh start`；状态：`bash ops/dashboard.sh status`。没有自动训练队列或新增定时任务。关闭网页不影响训练。

## 历史文件与依赖

`legacy/previous-platform` 保留此前五算法/OT平台源码，只用于追溯，其入口、预算、cost协议不属于当前baseline。当前入口不加载它。旧模型/日志和历史实验保留，避免丢失证据；无需复制或重跑旧方法。

两个上游目录 `cage-challenge-4/` 与 `Hierarchical-MARL/` 和运行环境放在服务器，不作为本仓库源文件上传。锁定版本及核对方法见 [部署说明](docs/PROJECT_LAYOUT_zh.md)。本仓库上传源码、指标与配置；完整模型包在服务器 `deliverables/cage4-baseline-20261023.tgz`，不混进Git历史。原始训练源码快照在 `evidence/`，新runner修改不追改历史证据。

沿用作者从state提取事件/block/decoy元数据的观测协议，不称纯Blue observation。引入新观测/约束需另外命名并公平比较。第三方源码与权重授权见 [NOTICE](NOTICE.md)。
