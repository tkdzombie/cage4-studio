# Baseline 实现审计（2026-10-20）

结论：当前是**可追溯的作者核心 + 明确的CAGE4适配**，不是空骨架或改名的简易Actor-Critic。但“代码能训练”“适配符合算法机制”“复现论文性能”是三个不同结论。前两项已有下述检查；最后一项尚未完成，不能把短程reward当作论文baseline成绩。

## 各算法定位

| 方法 | 实际核心 | 可在论文中准确使用的名称 | 限制 |
|---|---|---|---|
| IPPO | MAPPO作者R_MAPPO，critic用各自观测 | IPPO，作者PPO核心的CAGE4适配 | 不是单独的IPPO参考代码性能复现 |
| MAPPO | 作者R_MAPPOPolicy/R_MAPPO、集中critic | MAPPO，五独立策略版本 | 与作者默认共享策略不同，已固定为共同异构设置 |
| HAPPO | 官方on-policy仓库内HAPPO_Policy/HAPPO | HAPPO，作者核心的CAGE4适配 | 随机顺序、前序importance factor；不是HARL版本 |
| MAPPO-L | MACPO作者MAPPO-Lagrangian子项目 | MAPPO-L，作者核心的CAGE4适配 | 自定义业务cost和预算，不继承环境原生安全约束 |
| MACPO | 作者CPO约束步/回溯，替换离散分布KL与CG数值边界 | **MACPO-Discrete（CAGE4 adaptation）** | 原Safe-MuJoCo连续动作基准，不能声称原版未经修改或已有理论安全保证 |

Random/Sleep用于环境诊断；Monte Carlo Actor-Critic用于教学/消融；my_algorithm默认继承MAPPO，它们不补足上述正式论文比较组。当前未实现QMIX、MAT、H-MARL或新GNN算法，不能将名字写进“已实现baseline”表。

来源锁：on-policy `de66d7a4b23fac2513f56f96f73b3f5cb96695ac`，MACPO `b80a9f5b4a0049125a827be8fb9c477f69ae021b`。加载时拒绝commit或tracked源码变动；记录作者源码及LICENSE摘要。CybORG锁定 `8c3c50ca54b176c2de199847944e8dcc035497e3`，拒绝tracked变动，并记录全部Python源码摘要、实际依赖版本及本工程源码摘要。checkpoint评估检查这些来源和原配置；修改观测/算法后应重新训练，不静默复用旧权重。

## 核对与修正

- **策略梯度**：真实on-policy采样、旧log probability、合法动作mask、GAE、PPO clipping和作者optimizer；未使用随机动作假装训练。
- **多步动作**：CybORG会忽略执行中新增选择，现在IN_PROGRESS只允许Sleep；不把忽略的决策作为有效策略选择。
- **批次**：8个真实完整episode收集后更新；同批未更新策略。每个terminal将GAE递推截断，不让第二回合reward泄漏到第一回合。最后不满批次使用实际数据，不填充虚假步。
- **网络**：作者MLP先创建第一隐藏层，再追加layer_N层，因此正式两层256使用layer_N=1。没有把上游layer_N=2误称“两层”。默认lr5e-5、PPO10轮是共同起始profile，不是已调优最优参数。
- **CTDE**：IPPO critic只用自身，其他方法集中critic拼接5个Blue输入；actor只用自身。无全局感染真值，不向actor传Green业务明细。
- **顺序方法**：HAPPO、MACPO、MAPPO-L按作者separated runner方式，更新一名agent后累计联合importance factor。
- **安全量纲**：环境每步cost∈[0,1]，训练器比较批次内平均episode累计cost与rate×平均episode长度；避免把8回合总cost与1回合预算比较。研究者指标单独记录实际episode mean、usage、超限情况。
- **离散MACPO**：masked Categorical的精确KL(old||new)，保留new分布二阶梯度；CG处理零梯度并检查正曲率。CPO回溯拒绝更新是允许的，不能强行修改权重冒充成功。
- **诊断**：MACPO上游cost_loss日志错误地累加reward value_loss，已省略该错误字段；另算真实cost critic更新后MSE。MACPO没有普通PPO actor_loss，不能把KL或loss_improve假标成同一种loss。
- **评估**：显式场景seeds与训练、预算校准seeds分离；检查评估前后actor参数不变，保存逐seed分数、均值、样本标准差与超限率；代码不支持断点续训，未伪造resume。

## 安全方法的实质限制

作者cost critic使用gamma=.99的GAE，MACPO/MAPPO-L的约束surrogate还使用(1−gamma)缩放以及作者的cost advantage标准化；本工程保留这些作者规则。与此同时，CAGE4业务预算/评估采用有限回合**未折扣**平均业务失败成本。这是明确记录的实用适配，不能将surrogate满足解释为真实episode逐条不超预算的证明。需要在论文报告经验超限概率和严格/主/宽松预算敏感性；如研究目标要求精确有限时域理论保证，需要另行推导并验证对应surrogate，而不能仅更改gamma=1（会使作者(1−gamma)项退化）。

业务cost和Sleep参考预算是本工程定义，不是CAGE4官方SLA。Sleep均值可行也不证明每个测试场景可行。均值约束E[C]≤d与逐episode概率P(C>d)是不同指标，不能混用。

## 验证证据与复现

M3 macOS已有运行环境实测：全部五方法各2个真实20配置步训练episode、2个独立评估seed；5个actor和reward critic均实际变化，安全方法5个cost critic均变化，数值有限，checkpoint可加载，评估不更新权重。这只证明接口和更新链路。更完整horizon验收另见仓库本次变更的验证说明，绝不当作性能实验。

`paper/validation.py`还对每种方法的5个实际作者buffer/trainer做以下对照：旧logp重算一致；[1,1,0,1,0]终止mask；独立手算的两episode reward GAE；安全方法cost GAE、平均episode cost=.9、rate×2阈值换算。结合`tests/test_safety.py`的解析KL/Fisher/零梯度测试，以及`test_observations.py`的信息边界与时龄/扰动测试，检查关键适配而非仅检查程序不报错。

在远程研究项目终端执行（使用临时runs，避免挤掉正式最近5次）：

```bash
cd "$HOME/cage4-research"
export PYTHONPATH="$PWD/protocol:$PWD/cage4-baselines/src:$PWD/server/src"
PY="$HOME/cage4-research-env/bin/python"
"$PY" -m unittest discover -s tests -v
"$PY" -m unittest discover -s cage4-baselines/tests -v
"$PY" ops/validate_paper_baselines.py --runs /tmp/cage4-audit/runs \
  --upstreams "$HOME/cage4-upstreams" \
  --budget cage4-baselines/configs/safety/availability-v1.json \
  --output /tmp/cage4-audit/report.json
```

验收目录用完删除。报告不含个人凭据；实验/权重不提交GitHub。CI运行轻量核心测试，不下载大仿真/torch依赖，因此CI绿色不等于真实仿真或收敛验证。

## 距离正式论文结果还缺什么

按相同环境交互预算、同一观测与网络容量，对每方法至少5个训练seed，再用固定100个评估场景。调参只看独立验证集合，另设最终测试集合，不能反复根据最终测试seed挑模型。建议将现有53001–53100作为最终测试集，验证另用54001–54020；训练和预算校准范围已分离。当前评估CLI接受显式集合，调用方仍须遵守集合冻结规则。至少报告跨训练seed统计、不确定性与样本效率，不把一个checkpoint的100个场景当100次独立训练。

当前统一2000回合只是99.8万transition的起始预算，不保证足够收敛。完成以上实验后，才有证据评价baseline强度、观测改进或新约束算法的优势。本审计不预填论文结果，不给“论文级已认证”标签。

参考：[MAPPO作者实现](https://github.com/marlbenchmark/on-policy)、[MACPO/MAPPO-L作者实现](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)、[MACPO论文](https://arxiv.org/abs/2110.02793)。

完整horizon验收可在上述命令后加 `--full --algorithms mappo macpo`：使用提交的256双隐藏层与8回合完整批次，每方法3992条训练transition、2个独立完整评估episode；不开回放，避免验收生成大量临时文件。它仍是集成检查，不能证明2000回合训练已收敛。

本次完整horizon实测已通过：MAPPO和MACPO各8×499=3992条真实训练transition，只在第8回合更新，随后各在53001/53002完成499步评估；所有critic更新、数值检查及checkpoint读取通过。它不是多训练seed性能实验。另通过13项协议/服务与24项研究/真实环境单元测试，全部五方法的短程训练、评估和作者buffer数值对照均通过。

当前Sleep校准预算的统计可行性已验证，但尚未证明它对训练后的各方法都形成有效约束。正式研究应在独立验证场景检查预算是否过松/不可行并冻结预算设计，报告三档敏感性，避免在最终测试集上按结果挑预算。
