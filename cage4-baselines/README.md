# cage4-baselines · 远程 PyCharm 研究项目

本目录是独立研究项目。打开它、选择研究解释器、修改算法和 YAML，即可运行实验；网页只读取产生的记录。

## 先跑通

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines quickstart --runs "$HOME/cage4-experiments"
```

一键完成真实环境、Actor-Critic 训练、Random/Sleep 对照与三 seed 评估，结果可在网页查看。单独检查环境使用 `python -m cage4_baselines smoke`。完整用法见 [跑通指南](../docs/BASELINES_START_zh.md)。

## 找文件

```text
cage4-baselines/
├── pyproject.toml                 # 包依赖、Python 版本与 cage4 命令
├── environments/                 # 依赖版本参考、解释器说明
├── configs/
│   ├── train/                    # 训练参数
│   ├── evaluate/                 # checkpoint 评估参数
│   └── baselines/                # Random/Sleep 对照参数
├── src/cage4_baselines/
│   ├── core/types.py             # 策略输入数据契约
│   ├── envs/                     # CybORG 创建、适配、Blue observation 编码
│   ├── algorithms/               # 算法实现与注册
│   ├── experiments/runner.py     # 训练、评估、seed、checkpoint 调度
│   ├── recording/telemetry.py    # 研究者遥测，含 Ground Truth
│   ├── config.py                 # YAML 字段和范围检查
│   └── __main__.py               # python -m cage4_baselines
├── tests/                        # 研究项目测试
├── .run/                         # PyCharm 共享运行配置
└── artifacts/                    # 可选临时输出，不提交 Git
```

算法目录分别包含 actor_critic.py、random.py、sleep.py、registry.py、base.py。优先在这里实现新算法；不要到 Server 页面里改训练逻辑。顶层 adapter/engine/observations/runner/telemetry.py 仅为 v0.3.0 导入兼容入口，新代码使用上面的正式路径。

## 从本项目运行（远程终端）

先按仓库安装文档配置专用研究解释器并安装本地 editable 包。然后在本目录执行：

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines train \
  --config configs/train/actor_critic.yaml --runs "$HOME/cage4-experiments"

"$HOME/cage4-research-env/bin/python" -m cage4_baselines evaluate \
  --config configs/baselines/random.yaml --seeds 1001 1002 1003 \
  --runs "$HOME/cage4-experiments"
```

`cage4` console command 在重新安装 editable 包后也可使用，参数相同。旧 `python -m cage4_baselines.runner` 保持可用。Actor-Critic 评估必须额外指定真实 checkpoint，推荐使用 configs/evaluate/actor_critic.yaml，不更新权重。

完整说明见 [系统使用说明书](../docs/USER_MANUAL_zh.md)、[项目架构说明](../docs/PROJECT_STRUCTURE_zh.md) 和 [添加算法](../docs/RESEARCH_zh.md)。

## 论文算法（0.4.0）

IPPO / MAPPO / HAPPO / MAPPO-Lagrangian / MACPO 使用 `python -m cage4_baselines.paper`，配置位于 `configs/paper/`，业务预算位于 `configs/safety/`，作者核心通过外部锁定源码加载。环境安装仍沿用本项目解释器。先阅读 [完整算法使用说明](../docs/PAPER_BASELINES.md) 和 [业务 cost/预算依据](../docs/SAFETY_COST_zh.md)；MACPO 含明确的离散适配，不声称论文结果已复现。

最简单的 PyCharm 用法：打开根目录 `run_baseline.py`，选算法/轮数/seed，右键 Run。自己的算法写 `my_algorithm.py`，入口算法名改为 `my_algorithm`。见 [简明说明](../docs/START_SIMPLE_zh.md)。
