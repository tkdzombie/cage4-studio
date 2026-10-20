# 有约束baseline实验管理

用户已授权停止原MAPPO，自主调试、运行和管理有约束算法。本轮范围是当前实际实现的 **MACPO-Discrete、MAPPO-Lagrangian**；不重新训练IPPO/MAPPO/HAPPO，也不把自定义MAPPO模板冒充新的有约束算法。

## 固定协议与验收

- 环境、官方reward、Blue信息边界、cost定义保持锁定；主预算0.05，敏感性档0.03/0.10，不根据最终测试结果移动。
- 500配置步，实际499次transition；256双隐藏层、16回合冻结策略批次、16环境进程、8模型线程。两算法顺序运行，避免重复占满服务器。
- 每50回合保存，结束保存最后一次；不宣称支持断点续训。数据根目录仍按最近5次实验保留，先完成该模型评估再进入下一个任务。管理摘要另存，不在GitHub上传权重或原始实验数据。
- 主要评估随机策略（对应受约束的随机策略目标），同时完整报告确定性greedy结果，不能根据哪个成绩好再选择主模式。训练config保持不变，评估模式另存manifest/evaluation，严格核验源码、依赖、配置和budget SHA。
- 开发验证仅54001–54020；预算参考41001–41020、42001–42020；最终53001–53100不用于调参。smoke的短horizon结果不可与完整500步得分混用。
- MACPO原论文和作者仓库的任务为Safe Multi-Agent MuJoCo，不存在可以直接照搬的MACPO官方CAGE4回报。CAGE4公开强参考约−100至−200，但来自其他方法、观测和wrapper，不是本实现的保证值。先以完整开发验证均值≥−500且平均cost≤0.05为阶段目标，再争取−300附近及更好的可比表现；目标不是宣称“论文复现已完成”的自动标签。
- 达到阶段目标后仍需冻结实现/超参数，至少3个（正式建议5个）独立训练seed及100个最终评估场景，报告跨训练seed的统计、环境交互预算、成本与超限率。只达成本但reward差，或只达reward但成本超限，均不算目标达成。

参考：[MACPO/MAPPO-L作者仓库](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)、[CAGE4官方总结](https://ojs.aaai.org/index.php/AAAI/article/view/35158)。

## 当前阶段与后续有限调参

第一阶段每算法seed101训练2000回合；检查第200、1000、2000回合checkpoint，分别在相同20个开发场景以随机和greedy方式评估。中间checkpoint的评估在该次训练结束后执行，不干扰采样CPU预算。全部完成只标记`phase_complete`，由本聊天后续检查分析，不自动称为收敛。

诊断记录包括实际策略KL（MACPO拒绝候选步后的真实KL，而非仅候选KL）、actor是否更新、成本critic前后MSE、平均累计成本/预算、MAPPO-L乘子变化、采样与更新时间。MAPPO-L公开可配置`lamda_lagr`、`lagrangian_coef_rate`；MACPO公开`kl_threshold`、`fraction_coef`、`ls_step`，由验证与来源记录核验。作者cost optimizer共用`critic_lr`，不提供不起作用的独立cost_lr旋钮。MACPO每批实际一次CPO更新，YAML明确ppo_epoch=1。

作者MACPO日志还将lam/nu初始化为0却从未累加实际求解值；新版诊断删除这两个误导字段，改记求解器真实返回的`cpo_reward_dual`/`cpo_cost_dual`，不改变策略更新。已冻结的早期实验若仍有lam/nu=0，不能据此判断约束未生效；必须结合成本、实际KL、actor更新和后续版本的真实诊断。原始cost_loss别名错误仍按审计说明处理。

先判断数值和约束路径，再根据上述日志作单变量对照。MACPO若实际KL远小于阈值且更新过小，可将作者默认fraction_coef=0.1与1.0对照，仍保留KL/成本回溯接受检查；MAPPO-L可比较dual rate=0.0005与0.005。每种方法每次只改变一个因素，每个候选先用相同2000回合与seed101，最多再做6次候选实验；不得无限盲跑。选定后在相同seed102/103复核，再决定统一扩大到10000回合。任何更改观测、成本目标或训练机制必须单独命名、保存新源码并重新训练，不能将变体成绩偷换为原版算法复现。

若2000回合仍持续改善，不能仅因未达到−500就判算法失败；若训练和验证长期平坦，先检查KL/critic/乘子/观测与动作分布。多轮仍无改善、单任务超过6小时、或磁盘不足时保存证据并报告，不无限消耗资源。当前模型checkpoint较大；不自动启动5万回合任务，先按实测存储需求安排阶段预算。

## 服务器上的管理程序

研究项目的 `ops/constrained_campaign.py` 负责有界顺序队列，独立于本地网页客户端。启动前拒绝已有活动训练和重复管理器；冻结算法源码、协议模块、配置与预算；进程身份使用PID+启动时间+命令核对。状态原子写入control目录下的`state.json`，日志按任务拆分。遇到异常保留日志，继续检查另一种算法；遇到用户STOP、资源阈值则停止整个队列。管理程序不会自行宣称达标或改预算。

远程Mac Studio终端，首次启动示例（已有队列在运行时不要再执行）：

```bash
cd "$HOME/cage4-research"
export PYTHONPATH="$PWD/protocol:$PWD/cage4-baselines/src"
"$HOME/cage4-studio/.conda-env/bin/python" ops/constrained_campaign.py \
  --control "$HOME/cage4-experiment-control/constrained-initial" \
  --algorithms macpo mappo_lagrangian --seeds 101 --episodes 2000
```

control目录必须是新的；不会覆盖旧状态或伪装resume。私有control路径在部署时记录，本聊天通过SSH查看；公开仓库不包含用户名或IP。停止方式是在对应control目录创建`STOP`文件，管理程序向核对身份的训练父进程发送SIGINT并等待其清理worker；超过120秒才终止该任务所属进程组。正常停止保留已有checkpoint，不自动清空实验。

默认保持30GiB磁盘余量，每个训练/评估子任务最长6小时。评估原始回放只临时生成，逐seed回报、成本均值及超限率汇总保存后清理；正式训练仍保存完整回放。原始训练run按既有最近5次策略删除，管理摘要和源码版本便于追溯此前比较，但不能冒称已删除的旧checkpoint仍可下载。

当前聊天已设置每20分钟的定期跟进：检查队列、进程、数值、资源及结果，必要时修复并推进下一阶段，只在有意义的变化时通知。服务器队列不依赖本地客户端持续开启；本地定期跟进需要Codex和电脑保持运行，离线期间不承诺继续进行人工式代码诊断。用户停止授权后不得自动重启。

## 第一轮MACPO阶段记录与MAPPO-L修复

seed101、2000训练回合、20个固定开发场景，主随机策略结果如下（成本目标5%）：

| checkpoint | 随机回报均值 | 随机平均成本 | greedy回报均值 | greedy平均成本 |
|---|---:|---:|---:|---:|
| 200 | -5250.60 | 18.23% | -6764.80 | 19.84% |
| 1000 | -4968.50 | 17.18% | -6499.65 | 19.25% |
| 2000 | -3997.65 | 13.39% | -6596.25 | 19.46% |

所有检查点两种模式的20个场景均超预算。MACPO有缓慢改善，但未达到阶段目标，单训练seed不能证明论文性能。逐seed成本和回报在服务器私有control保存，公开文档仅保留汇总。

随后发现作者MAPPO-L的Categorical entropy与active_masks广播错误，确实放大熵梯度。受影响试跑已停止，保留记录但不得作为正常baseline性能结果。修复详见BASELINE_AUDIT_zh.md；改后用新control和独立run仅重跑MAPPO-L，不重复已完成MACPO。原始constrained-initial-v2因此停止；后续以聊天记录和新的control状态为准，不重启旧队列。

## 修复版MAPPO-L开发验证及第一轮单因素对照

修复版seed101的2000回合训练与全部开发验证已完成，仍按随机主模式、greedy辅模式完整报告：

| checkpoint | 随机回报均值 | 随机平均成本 | 随机超限场景比例 | greedy回报均值 | greedy平均成本 |
|---|---:|---:|---:|---:|---:|
| 200 | -3938.95 | 13.35% | 100% | -6874.05 | 20.49% |
| 1000 | -1089.40 | 2.50% | 10% | -5967.95 | 17.26% |
| 2000 | -846.10 | 1.59% | 0% | -6799.55 | 20.30% |

均为相同20个开发场景、同一训练seed。成本满足只说明这次开发验证的经验结果；0/20不代表真实超限概率为零。回报仍低于−500阶段目标，greedy也明显不达标，不能隐藏或据此声称已复现论文性能。

下一轮保持2000回合、seed101和全部协议，仅MACPO的fraction_coef从0.1改0.5、MAPPO-L的actor lr从5e-5改1e-4，顺序执行并各评估200/1000/2000。这是最多6次候选预算中的前2次。MACPO中间步长由已保存的隔离训练批次诊断支持：1.0会产生大的importance ratio尾部，因此先检验0.5；MAPPO-L学习率对照检验相同交互预算下的改善速度，不改变约束或熵系数。未根据最终测试集选择参数。

管理器支持`--overrides candidate.json`，schema为`cage4.campaign-overrides/1`，例如：

```json
{"schema":"cage4.campaign-overrides/1","algorithms":{"macpo":{"fraction_coef":0.5},"mappo_lagrangian":{"lr":0.0001}}}
```

每算法最多一个受支持超参数，拒绝预算/环境/seed等协议覆盖；范围在启动前验证。原文件规范化保存到新control的overrides.json，实际合并config写入task/run/checkpoint，评估仍严格核验同一实际config。默认YAML不变，旧control不修改、不resume。MACPO原轮与本轮的代码差异只涉及未使用的MAPPO-L分支及正确日志诊断，CPO更新机制保持相同；所有来源各自保留。
