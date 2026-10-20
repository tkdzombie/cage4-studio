# CAGE4 研究项目

在 Toolbox 的**远程 PyCharm**中使用本目录。环境和依赖已在现有 Mac Studio 配好。

## 跑已有 baseline

打开 `src/cage4_baselines/experiments/run_baseline.py`，顶部设置：

```python
ALGORITHM = "mappo"
EPISODES = 2
SEED = 101
```

右键这个文件 → Run；需要断点时选 Debug。不需要再填运行参数或工作目录。

可选：`ippo` / `mappo` / `happo` / `mappo_lagrangian` / `macpo`。

在远程终端中，先进入上述 experiments 文件夹，再执行 `python run_baseline.py`；python须为已配置的服务器解释器。现有 Conda Python、独立研究 Python均可。

## 写自己的算法

编辑 `src/cage4_baselines/algorithms/my_algorithm.py`，保留类接口，在这里定制网络、训练器、损失或更新逻辑。模板默认沿用 MAPPO，未修改时不算新的算法。

启动时将 `ALGORITHM = "my_algorithm"`，仍运行同一个 `experiments/run_baseline.py`。设为 `mappo` 会运行原有 MAPPO，不会运行你写的文件。

算法参数、学习率等在 `configs/paper/`；日常轮数与seed优先使用启动文件顶部设置。只有研究确实改变任务、观测或业务cost时，才改环境/cost并重新验证，不必为换算法修改网页。

## 目录只按职责找

```text
cage4-baselines/
├── configs/paper/                    # 五算法参数
├── configs/safety/                   # 已校准的业务预算
├── configs/train,evaluate,baselines/  # 基础 Actor-Critic 与 Random/Sleep
├── environments/                     # Python依赖版本与环境说明
├── src/cage4_baselines/
│   ├── experiments/run_baseline.py   # 日常运行这个文件
│   ├── algorithms/my_algorithm.py    # 自己的算法写这里
│   ├── envs/                         # CybORG环境与Blue观测编码
│   ├── paper/                        # 作者算法接入与统一训练/评估
│   ├── safety/                       # 业务cost与预算校准
│   ├── recording/                    # 回放和遥测
│   └── core/                         # 输入数据契约
├── tests/                            # 回归测试
└── pyproject.toml                    # 包依赖
```

不在源码中保存虚拟环境、第三方仓库、实验权重或临时压缩包；实验写入 `~/cage4-experiments`。旧的平铺重复YAML和手工 .run 配置已移除，少量旧导入薄兼容入口保留，避免破坏已用脚本。

App随时可打开看已有记录；训练第一个完整回合结束后出现新曲线。网页关闭不停止训练。

详细资料：[简明说明](../docs/START_SIMPLE_zh.md) · [算法来源与正式评估](../docs/PAPER_BASELINES.md) · [cost与预算](../docs/SAFETY_COST_zh.md) · [安装和回滚](../docs/INSTALL_zh.md)。
