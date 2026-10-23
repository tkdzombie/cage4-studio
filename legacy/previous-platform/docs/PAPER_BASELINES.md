# 论文比较算法与统一实验协议

本次新增五个实际调用作者代码、能够训练和加载权重评估的算法入口。Random、Sleep 和原 Monte Carlo Actor-Critic 保留为环境检查/消融工具，不再把它们当作充分的论文比较组。短时验收证明代码链路可运行，不证明算法收敛、排名或安全保证。

## 算法来源与适配边界

| 名称 | 上游核心 | 本工程适配 | 信息与训练方式 |
|---|---|---|---|
| IPPO | [MAPPO 作者仓库](https://github.com/marlbenchmark/on-policy) R_MAPPOPolicy + R_MAPPO | critic 使用各自 Blue 观测；原 PPO clipping、GAE、minibatch 更新 | 五个独立 actor/critic |
| MAPPO | 同上 | critic 输入五个 Blue 观测的拼接 | CTDE；执行 actor 只用自身观测 |
| HAPPO | 同一官方 on-policy 仓库的 HAPPO_Policy + HAPPO | 随机顺序更新，累计前序 agent 的 importance factor | 异构独立策略，集中 critic |
| MAPPO-Lagrangian | [MACPO 作者仓库](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) MAPPO-Lagrangian 子项目 | 共同业务 cost、cost critic、作者乘子更新，顺序 importance factor | 集中 reward/cost critic |
| MACPO | 同一作者仓库 MACPO 子项目 | **离散动作移植**：masked Categorical KL、零梯度 CG 处理；作者约束步与回溯保留 | 集中 reward/cost critic；CPU |

这里的 HAPPO 来自官方 MAPPO 仓库内的实现，不冒称采用另一套 HARL 代码。MACPO 原基准是 Safe Multi-Agent MuJoCo，不能直接把高斯连续动作 KL 用于 CAGE4。我们的移植与上游差异集中在 `paper/discrete_macpo.py`，并用解析 KL、Fisher 曲率、零 cost 梯度测试和真实仿真验证。未声称继承论文中每轮约束满足的理论保证。

上游锁定在 `paper/upstreams.json`：on-policy `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`；MACPO `b80a9f5b4a0049125a827be8fb9c477f69ae021b`。启动检查 commit 和 tracked 文件是否改动，记录 Python 源码/许可证 SHA256。使用 namespace root 避免上游包初始化时加载无关 runner、MuJoCo 和 WandB；算法模块本身保持原文件。源码独立放在缓存目录，不安装整个上游项目。

on-policy 使用 MIT。MACPO 锁定版本的 LICENSE 含未解决的合并标记；本工程没有擅自修复或重新声明其许可，也不把作者源码打进我们的发布包。若要再次分发这些作者源码，先向上游澄清。我们自有仓库的许可仍按 NOTICE 处理。

## 环境与统一输入

- 使用同一 CybORG、FiniteStateRedAgent、五个 Blue、官方 BlueFixedActionWrapper 合法动作掩码。
- actor 只接收本 agent 的 Blue observation 结构化特征 `blue-belief-slots/1`，布局与信息边界见 [OBSERVATIONS_zh.md](OBSERVATIONS_zh.md)；不读取网络 Ground Truth、其他 Blue 输入、红队 Session 或 researcher cost 明细。
- 集中 critic 仅拼接 Blue 观测，不使用 Ground Truth。safe learner 收到一个共同团队 cost 标量，五个 cost critic 使用同一团队约束；不是五个不同的辖区预算。
- 全部五策略、不共享参数、feedforward CPU、两层256 隐藏单元、`layer_N=1`（上游额外层数）、gamma=.99、GAE lambda=.95、PPO epoch=10、minibatch=1、Adam lr=5e-5、clip=.2、entropy=.001。MACPO 为作者 trust-region 单次更新，epoch 强制为 1；不能用 PPO epoch 伪装 MACPO 更新次数。
- 全部 reward 乘 .01 进入训练，保存和评估仍为官方原始 reward；cost 每步在 [0,1]，不做 reward scaling。
- 关闭 PopArt/ValueNorm、RNN、人工 time-limit bootstrap。每批16个完整回合更新，GAE在每个terminal截断，buffer 使用真实执行的 T，绝不填充未执行步。作者默认值和显式覆盖的最终参数写入 `upstream-arguments.json`。

这是一套**控制共同输入的 CAGE4 比较协议**，不是声称逐项复现某篇 CAGE4 论文。尤其结构化编码、无 Ground Truth 的集中 critic、训练 batch 大小可能不同。论文应报告编码与信息权限；复现特定论文还需按该论文的代码/配置另建 profile。当前没有把 QMIX、H-MARL 或未核验的论文代码登记成已实现算法。

## 业务 cost 与预算

完整依据、公式、独立校准数据、三档预算见 [SAFETY_COST_zh.md](SAFETY_COST_zh.md)。新实验默认主预算为 0.05（严格0.03、宽松0.10；目标可行性尚未确立），500 配置步在锁定 CybORG 版本实际执行 499 次 transition。算法比较批次内平均episode cost，预算是 `rate × 平均episode长度`，不是直接将 rate 当累计 cost 阈值。

## 在远程 Mac Studio / PyCharm 终端运行

在研究项目的终端中执行。已有 Client 无需重新配置，网页依旧只读实验文件。

```bash
cd "$HOME/cage4-research"
PY="$HOME/cage4-research-env/bin/python"
export PYTHONPATH="$PWD/protocol:$PWD/cage4-baselines/src"

# 仅下载锁定作者源码，不安装 MuJoCo、WandB、Ray 等大依赖。
"$PY" -m cage4_baselines.paper.upstreams --root "$HOME/cage4-upstreams"

# 验证已经安装的真实环境及全部五算法训练/评估。
"$PY" ops/validate_paper_baselines.py \
  --runs /tmp/cage4-validation/runs --upstreams "$HOME/cage4-upstreams" \
  --budget cage4-baselines/configs/safety/availability-target-v2.json \
  --output /tmp/cage4-validation/report.json

# 正式训练（2000 episode；这是起始预算，不代表已收敛）。
"$PY" -m cage4_baselines.paper train \
  --config cage4-baselines/configs/paper/mappo.yaml \
  --budget cage4-baselines/configs/safety/availability-target-v2.json \
  --upstreams "$HOME/cage4-upstreams" --runs "$HOME/cage4-experiments"

# 使用该次运行保存的实际JSON配置与checkpoint做独立评估。
# <RUN> 替换成上一步输出的目录名；--episode选择已存在的权重。
"$PY" ops/evaluate_baseline.py \
  --run "$HOME/cage4-experiments/<RUN>" --episode 2000 \
  --upstreams "$HOME/cage4-upstreams" --runs "$HOME/cage4-experiments" \
  --seeds $(seq 53001 53100)
```

运行其他方法只需选择其 YAML；不要只改名字却继续加载另一个算法权重。PyCharm 的 Python module 设置为 `cage4_baselines.paper`，参数与上面一致；工作目录设为仓库根，或将配置路径改成项目内路径。PyCharm 参数框使用实际远程绝对路径，不依赖 shell 的 `$HOME`/`$(seq ...)` 展开。

配置修改建议复制到个人 profile，保留提交到论文的最终 YAML。训练 seed 使用 101/102/103/104/105，各跑一次；实际训练 episode seeds 采用独立范围 `100000 + seed*100000 + episode_index`。校准 41001–41020、验证 42001–42020、最终评估 53001–53100 分离；评估入口拒绝重叠校准 seed 或 checkpoint 内的训练 seed。

评估每个显式seed一个episode，不修改策略。动作模式由训练配置记录：本轮有约束MACPO/MAPPO-L统一deterministic_eval=false，以随机策略作为主评估，并另报greedy结果；其他历史profile保留自身设置。新版独立评估入口可用--action-mode显式报告两种模式，不改checkpoint中的原始配置或放宽来源检查，模式写入manifest和evaluation.json。正式比较应按相同环境交互预算、相同训练 seeds 和同一评估集合，报告跨训练 seed 的均值/标准差及 cost/超限率，不能把同一模型的 100 个测试场景当成 100 次独立训练。推荐起始训练预算每 seed 2000×499=998000 transitions；根据验证集曲线判断是否需延长，所有方法统一预算。

`--smoke` 仅允许缩短 horizon 做接口检查，manifest 会明确标记；20 步没有完整任务阶段，不可用于论文 reward/cost 比较。当前支持保存 actor、reward/cost critic、optimizers、乘子及配置供评估；**不支持断点续训**，未假装恢复 RNG/历史 buffer。

## 回报与网页解释

- `official_team_return`：每一步五个共享 Blue reward 取均值后累计，包含实际执行的终止步，用作本协议主要回报。
- `official_script_compat_return`：对照锁定官方 evaluation 脚本的“遇到 done 先 break”顺序，排除最后一次终止 transition reward。这个字段不意味着提交到排行榜已被验证。
- `team_return`：保留旧版五 Agent 回报之和，通常是前者的五倍，避免破坏旧读取器。

新 Server 单次曲线优先使用 official_team_return；对比页需要选择同一回报口径，旧记录没有该字段时不混入。actor_loss/value_loss/entropy 使用标准展示别名，并保留原始作者日志。MACPO 的上游 cost_loss 日志实际重复了 reward value_loss，因此不展示这个错误字段；另外记录更新后的真实 cost value MSE。MACPO 回溯拒绝 actor 步是合法行为，权重未变会如实记录，不伪造更新。

## 文件架构

| 位置 | 放什么 |
|---|---|
| `envs/` | 唯一真实仿真适配与 Blue observation 编码 |
| `configs/paper/` | 五个算法的统一训练 profile |
| `configs/safety/availability-target-v2.json` | 新研究目标5%/3%/10%，保留参考测量和可行性状态 |
| `configs/safety/availability-v1.json` | 历史校准预算及原始参考数据，旧实验继续使用 |
| `safety/costs.py` | 业务事件到标量 cost；与 actor 输入隔离 |
| `safety/calibrate.py` | 固定参考策略和独立 seed 校准 |
| `paper/upstreams.json`、`upstreams.py` | 作者源码版本锁、完整性与加载 |
| `paper/backend.py` | Rollout 到各作者 buffer 的桥接、顺序 factor、checkpoint |
| `paper/discrete_macpo.py` | 明确且可审查的离散 MACPO 适配 |
| `paper/__main__.py` | 统一 train/evaluate 与原子记录 |
| `$HOME/cage4-upstreams/` | 外部作者源码，不混入自有 Git 仓库 |
| `$HOME/cage4-experiments/` | 运行数据和权重，绝不提交 GitHub |

扩展新算法：先核验论文/作者仓库/许可证和 commit → 新增独立 backend（或复用已证明兼容的 buffer）→ 保持局部 actor 信息权限和 cost/reward 口径 → 添加作者源 provenance → 通过动作 mask、真实梯度、权重保存/读取、评估不更新与数值测试 → 再加入算法表。不靠注册名称宣称实现。

参考：[MAPPO 作者代码](https://github.com/marlbenchmark/on-policy)、[HAPPO 论文](https://arxiv.org/abs/2109.11251)、[MACPO/MAPPO-L 论文](https://arxiv.org/abs/2110.02793)、[CAGE4 官方环境](https://github.com/cage-challenge/cage-challenge-4)、[CAGE4 H-MARL 研究](https://arxiv.org/abs/2410.17351)。

默认磁盘仅保留最近5次运行，网页仅显示最新一次；自动清理和结构化输入版本见 [OBSERVATIONS_zh.md](OBSERVATIONS_zh.md)。验收请使用临时目录，不能占用正式实验保留名额。

若通过run_baseline.py覆盖了轮数/seed，评估应使用ops/evaluate_baseline.py读取该次运行保存的config.json，而不是未经覆盖的原始YAML。两者配置不同会被严格校验拒绝。

记录配置的JSON科学计数法必须用json.loads读取，避免YAML 1.1将5e-05解析为字符串。直接模块CLI的--config适用于YAML；记录配置的评估统一使用上述脚本，不修改正在运行的策略源码或放宽来源校验。

训练回合数、return口径与官方参考见 [TRAINING_TARGETS_zh.md](TRAINING_TARGETS_zh.md)。新训练保存`source/cage4_baselines/`，独立评估入口优先使用该次运行的源码快照；仍严格核验依赖版本、源码、配置和预算SHA，不允许用新预算评估旧权重并伪称原协议结果。快照是科研溯源数据，随整次实验按保留策略清理。

用户授权的有约束算法托管计划、资源限制和阶段验收见 [CONSTRAINED_EXPERIMENTS_zh.md](CONSTRAINED_EXPERIMENTS_zh.md)。
