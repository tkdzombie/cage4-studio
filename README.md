# CAGE4 Studio

独立算法研究项目 + 只读实验网页 + 本地 SSH 客户端。服务器已有环境可直接使用。

## 开始训练

在 Toolbox 的远程 PyCharm 中展开：

```text
cage4-research/cage4-baselines/src/cage4_baselines/
├── experiments/run_baseline.py   # 右键 Run / Debug
└── algorithms/my_algorithm.py    # 自己的网络、损失、更新逻辑
```

启动文件顶部只需设置：

```python
ALGORITHM = "mappo"
EPISODES = None
SEED = 101
```

现有算法：`ippo`、`mappo`、`happo`、`mappo_lagrangian`、`macpo`。跑自己的代码设置 `ALGORITHM = "my_algorithm"`；模板默认沿用 MAPPO，未修改时不算新算法。环境、配置、预算、实验保存路径自动定位，无需填写额外参数。终端在启动文件所在目录、使用现有服务器 Python 执行 `python run_baseline.py` 也可。

打开现有 Client.app 查看记录；网页可先开，关闭不影响训练。首个同步采样工作块完成后显示新曲线；并行采样期间可查看进程数、阶段和速度。

## 项目目录

| 目录 | 用途 |
|---|---|
| cage4-baselines/src/cage4_baselines/envs/ | CybORG 场景、step/reset、Blue观测编码 |
| cage4-baselines/src/cage4_baselines/algorithms/ | 自定义算法与基础对照 |
| cage4-baselines/src/cage4_baselines/experiments/ | 运行入口与基础实验流程 |
| cage4-baselines/src/cage4_baselines/paper/ | 五算法作者核心接入、统一训练与评估 |
| cage4-baselines/src/cage4_baselines/safety/ | 业务cost与预算校准 |
| cage4-baselines/configs/ | 按用途组织的有效YAML与固定预算 |
| cage4-baselines/environments/ | 依赖版本参考；实际环境放源码外 |
| protocol/ | 原子实验文件协议、状态与PID身份 |
| server/ | 只读图表、拓扑、动作/观测回放、实验比较 |
| client/ | 本地 SSH 转发与网页入口 |
| ops/ | 服务安装/停止/查询/回滚、验证与打包 |
| tests/、cage4-baselines/tests/ | 协议、服务、环境与算法回归测试 |

外部目录分别保存运行环境、锁定作者源码、实验数据和服务部署版本，不能当成临时文件删除。网页不执行训练，不读取checkpoint；Ground Truth和红队详细信息不输入Blue actor。

## 使用和实验协议

- [日常最简用法](docs/START_SIMPLE_zh.md)
- [作者来源、超参数、正式训练与独立评估](docs/PAPER_BASELINES.md)
- [业务cost、三档预算及独立校准依据](docs/SAFETY_COST_zh.md)
- [安装、服务管理和回滚](docs/INSTALL_zh.md)
- [实验记录格式与信息边界](docs/PROTOCOL.md)

MACPO是作者核心的离散动作适配，不冒称未修改的官方CAGE4版本。默认业务cost预算24.748%，由20校准+20独立验证场景确定，是本项目研究协议。五算法已验证真实训练、权重保存及独立评估；未声称论文规模收敛/性能或理论安全保证，当前不支持断点续训。第三方许可与本仓库授权边界见 [NOTICE](NOTICE.md)。

历史源码、旧入口、交付过程报告和重复说明不放在当前项目中；需要追溯时查Git历史。日常启动入口关闭字节码缓存写入，实验结果保存在源码目录外。

结构化输入、正式训练批次和最近5次运行的永久删除规则见 [观测与训练协议](docs/OBSERVATIONS_zh.md)。

研究实现核验见 [Baseline审计](docs/BASELINE_AUDIT_zh.md)：明确作者核心、CAGE4适配和仍需完成的正式多seed实验，不能将短程验收当作论文性能复现。

Mac Studio 多核参数、50回合checkpoint及可复现吞吐测试见 [多核训练](docs/PERFORMANCE_zh.md)。MAPPO支持同步多环境采样；自定义算法保留单进程断点调试。
