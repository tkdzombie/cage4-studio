# CAGE4 算法实验

**目前1种baseline：平坦独立PPO（五个Blue各自训练策略）。** 作者权重初始化、随机初始化是两种起点，不是两种算法。主对照采用从零训练的PPO。

服务器：`/Users/tiankedi/cage4-research`；PyCharm解释器：`.venv/bin/python`。

```text
run.py                   统一启动入口，PyCharm可直接Run
configs/                 只改实验配置：算法、初始化、预算、运行名
algorithms/              只改算法：ppo.py 与 my_algorithm.py
cage4/                   共用实现，一般不必改
  environment.py         官方CAGE4适配与固定观测/动作协议
  training.py            采样、更新、模型保存、结束后评估
  evaluation.py          加载模型，用相同场景与随机数评估
  config.py              配置验证
  dashboard.py           只读实时看板
models/                  服务器保留的可用PPO最终模型
runs/                    新实验的配置、源码、曲线和checkpoint
ops/、client/            看板与Mac客户端
```

`Hierarchical-MARL/`、`cage-challenge-4/`、`.venv-independent/`是服务器依赖目录。`runtime-freeze.txt`与`requirements.txt`记录依赖。不要编辑依赖来修改自己的算法。

## 常用命令

```bash
# 代表性baseline：随机初始化，seed321，150更新，24采样worker
.venv/bin/python run.py train --config configs/ppo.json
# 自己的算法（先改algorithms/my_algorithm.py，再改唯一run名）
.venv/bin/python run.py train --config configs/my_algorithm.json
# 评估保留的模型
.venv/bin/python run.py evaluate --run models/ppo-author --endpoints final
# 看板
bash ops/dashboard.sh start
```

PyCharm直接Run `run.py`默认读取`configs/my_algorithm.json`；也可在Run配置添加上述参数。新run拒绝覆盖已有目录。配置中的初始化为`scratch`或`author`；切换作者初始化时应合理设置lr/epochs，并对比较双方保持一致。

当前长训练固定150更新，约1796400环境transition（24worker时每次实际11976，完整回合采样）；不因短期横盘提前停止。非有限learner数值会停止；Ctrl+C或在当前run下创建STOP文件会在当前更新后保存末尾并停止。固定预算只表示训练充分程度，不自动宣称收敛。

每10更新保存latest，并保留initial/best/final；正式比较使用固定预算final。预算正常结束后自动在四个开发场景54001–54004评估initial/final并写入当前run，网页自动读取。单训练seed不等同跨seed稳定性。这些是新实验所需输出，旧测试/历史结果不再放项目中。

[算法修改与比较约定](docs/ALGORITHM_GUIDE_zh.md)。环境沿用作者state-derived事件/block/decoy观测，包含流量控制、官方原始奖励，500配置步/499实际transition。保存模型的旧`primitive_env`模块名已兼容，可继续评估两份既有模型。来源授权见NOTICE.md。
