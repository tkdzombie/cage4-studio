# 验证记录（2026-10-20）

## 实际运行结果

在用户 Mac Studio M3 Ultra 96GB / macOS 的现有 Python 3.10.22 环境，以及新建的独立 `cage4-research-env` 中完成验证。没有重新下载大型依赖。

- 12 项单元测试全部 PASS：原子写入失败保留旧记录、并发读取完整记录、legacy 部分行、PID 身份复用、路径/符号链接边界、合法动作掩码、未知 MAPPO 拒绝、观测稳定编码、五 Agent 权重更新和评估不更新、launchd 异步卸载回归等。
- 真实 CybORG Actor-Critic 训练 2 Episode × 20 步上限，比较两个 checkpoint：5 个 Blue 模型都发生权重变化。每个 Episode 保存 reward、actor loss、critic loss、entropy 和 checkpoint。
- checkpoint 在 seeds 1001/1002/1003 上独立评估，通过其 SHA256 不变确认评估未改写权重；Random/Sleep 在同一组 seed 上实际运行。
- 此 seed 的初始拓扑实际为 97 hosts / 96 interface links。不能固定要求所有 seed 都等于历史 smoke test 的 93/92。
- 所有执行过的 Blue 动作索引符合官方 mask。每步记录真实 Blue observation、红队动作/observation/Session；JSON Schema 验证 PASS。
- Streamlit AppTest 对真实记录执行实际 app，未出现异常；新旧格式均有只读适配。UI 的分项 loss 来自训练输出，不生成假曲线。
- Mac Studio launchd 实际安装、旧版回滚健康检查、重新安装新版均 PASS。修复了 bootout 返回早于任务消失的竞态；异常安装也实际自动恢复了旧服务。
- 本地 macOS `.app` 构建与 plist 检查通过；实际打开后建立受控 SSH 隧道，本地 18501 的健康请求通过。旧 8501 隧道保留。

完整验收脚本是 ops/validate_simulator.py。执行前设置 PYTHONPATH 为 protocol:server/src:cage4-baselines/src；--output 指定独立目录。完整实验数据保留在用户服务器上，不提交 GitHub。交付的独立 validation.json 为用户验收记录，不作为算法性能结论。

服务源码版本目录：`$HOME/cage4-studio-releases/CAGE4_Studio_v0.3.0`。
研究工作副本：`$HOME/cage4-research/cage4-baselines`。
研究解释器：`$HOME/cage4-research-env/bin/python`。
可恢复旧 dashboard 的已验证备份：`$HOME/cage4-studio-backups/<install-output-backup>`。
旧源码备份：`$HOME/cage4-studio-backups/source-v0.2.0-<date>.tgz`。
Server 日志：`~/cage4-studio-logs/dashboard.{out,err}.log`。

## 兼容性与未验证项

实际关键版本见 requirements-tested.txt；上游 commit 为 8c3c50ca54b176c2de199847944e8dcc035497e3。原 requirements-legacy.txt 要求 Arrow 14.0.2，而当前实际 Arrow 25.0.1；没有为形式上的 pin 对齐替换已运行环境。setuptools 为 80.9.0，避免 pkg_resources 缺失问题。旧 Gym 的维护状态警告仍存在。

原环境 `pip check` 并非全绿：CybORG 全量安装元数据还要求 paramiko、pytest、pytest-mock、sb3-contrib、sshtunnel、torch-geometric，当前缺少；lz4 和 tensorboardX 也不同于其精确声明。本次研究路径没有用到缺少的模块，真实 smoke/train/evaluate 已通过。不应把这解释为所有上游扩展功能都可运行，也未为此下载大型依赖。

CI 在 macOS 上运行轻量编译、shell 检查及核心单元测试，只安装 NumPy/PyYAML，主动跳过需要 Torch 的权重测试；M3 实机的上述全量测试补充该限制。分支与 PR 的两次 macOS CI 已通过（PR #1）；后续提交状态以实际 PR 检查为准。

没有验证：MPS、长时间收敛、百万步记录性能、其他 macOS 机器空白安装、重启后未登录启动、JetBrains Toolbox 的 GUI 各版本自动导入运行配置、代码签名/公证。Toolbox 的项目和解释器已准备好，但没有声称已替用户点击完成 PyCharm GUI。当前不支持断点续训、IPPO、MAPPO、集群调度或进程级恶意插件沙箱。
