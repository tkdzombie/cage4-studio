# 基础环境与 Baseline：先跑起来

现有 Mac Studio 研究环境已经安装依赖，下面直接使用远程研究解释器，不重新下载大依赖。所有命令都在 **远程 Mac Studio 终端** 或 PyCharm 的远程 Terminal 执行。

## 1. 一条命令完成整个基础实验

```bash
cd "$HOME/cage4-research/cage4-baselines"
"$HOME/cage4-research-env/bin/python" -m cage4_baselines quickstart \
  --runs "$HOME/cage4-experiments"
```

会依次执行：

1. 实际创建 CybORG，完成一个合法动作的环境 Episode。
2. 用五个独立 Actor-Critic 策略训练 2 Episode，每个 Episode 最多 20 步，保存真实指标和 checkpoint。
3. 比较两次 checkpoint，确认五个网络都更新了权重。
4. 在 seeds 1001、1002、1003 上分别评估训练模型、Random、Sleep，每个 seed 1 Episode。
5. 验证评估没有改写 checkpoint，保存汇总报告。

训练/评估的四个独立 run 直接放进 --runs 目录，Studio 网页刷新后即可选择查看。汇总报告路径会打印在终端，位置为 `<runs>/starter_suites/quickstart_<唯一标识>/report.json`；suite 目录本身不作为模型 run 出现在网页列表中。

这是短程可运行性演示，不代表收敛或科研性能结论。正式实验使用 train/evaluate 命令和单独 YAML。需要不同评估种子可传 `--seeds 11 22 33`。`--config` 可指定自定义的 Actor-Critic YAML，至少 2 Episode；该入口会把 checkpoint_every 设为 1，确保能检验权重更新，实际配置仍保存在 run/config.json。

## 2. 只检查环境

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines smoke --seed 7629 --steps 20
```

输出五个 Blue 的 observation 维度、动作数量、合法动作数量、实际步数及回报。基于真实 CybORG 执行，不生成模拟结果。

基础接口代码在 [envs/adapter.py](../cage4-baselines/src/cage4_baselines/envs/adapter.py)，可以这样使用：

```python
from cage4_baselines.envs.adapter import Cage4Adapter
from cage4_baselines.algorithms import create

env = Cage4Adapter(seed=7629, steps=20)
inputs = env.reset()
policy = create('random', inputs, {'seed': 7629, 'plugins': []})

while True:
    actions = policy.act(inputs)
    inputs, rewards, terminated, truncated = env.step(actions)
    print(rewards)
    if not inputs or all(
        terminated.get(a, False) or truncated.get(a, False) for a in actions
    ):
        break
```

reset/step 自动管理当前输入状态，无需调用其他同步方法。step 要求每个 active Blue 都有一个动作，拒绝遗漏、越界、被 mask 禁止的动作；支持 Python int 和 NumPy integer，不把布尔值当动作。结束后须 reset 才能开始下一回合。

可直接运行的完整示例：

```bash
"$HOME/cage4-research-env/bin/python" examples/environment_rollout.py \
  --algorithm random --seed 7629 --steps 20
```

将 random 改为 sleep 即可运行 Sleep 环境回放。这个小例子只在终端打印结果；需要网页记录时使用 train/evaluate 或 quickstart。

## 3. 三种 Baseline 的实际实现

| Baseline | 实现文件 | 行为 |
|---|---|---|
| Random | algorithms/random.py | 在当前官方合法掩码内随机选动作 |
| Sleep | algorithms/sleep.py | 使用合法的官方 Sleep 动作 |
| Actor-Critic | algorithms/actor_critic.py | 五个独立可学习网络，真实反向传播与 Adam 更新 |

Actor-Critic 的每个网络为 128×128 隐藏层，输出 action logits 与 value；损失由 REINFORCE actor loss、0.5 MSE critic loss 和 entropy bonus 构成。采用折扣 Monte Carlo return 与梯度裁剪。它是基础参考，不是 PPO/IPPO/MAPPO。

策略只接收本 agent 的真实 observation 编码、官方动作 mask 与动作标签；全局拓扑及红队遥测不进入策略。编码及局限见 PROTOCOL.md。

## 4. 正式训练与多 seed 评估

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines train \
  --config configs/train/actor_critic.yaml --runs "$HOME/cage4-experiments"
```

默认 30 Episode / 100 步上限。完整评估与对照命令见 [系统说明书](USER_MANUAL_zh.md)。评估必须指定真实 checkpoint；只有 Actor-Critic 评估需要它，Random/Sleep 不传 checkpoint。

数据按 cage4.run/1 原子记录。训练日志指标、逐步观察、拓扑、状态、模型参数分别保存在独立文件中；checkpoints 不能当作断点续训入口，本版没有 resume。

## 5. 运行与测试状态

新增基础环境接口及 smoke/quickstart 后，Mac Studio 实机仓库级 8 项 + 研究项目 11 项，共 19 项测试通过；真实一键套件完成，并检验五个模型权重变化、三 seed 评估及 checkpoint 不变。轻量 CI 不安装 CybORG/Torch，会显式跳过相应实机测试。

旧 `python -m cage4_baselines.runner train/evaluate ...` 和 inputs() 兼容接口保留，新的示例使用统一模块入口与正常 reset/step。Server 固定发布目录和既有实验数据无需重新安装或迁移。
