# OT连续性baseline适配与有界试测

日期：2026-10-21。用户已授权推进至baseline开始测试。仅运行有约束算法；官方模拟器、reward、旧可用性cost与旧实验均保持原版本。

## 当前状态

主任务资格推进更新：合法Analyse巡检规则在冻结后的128个新场景中平均−291.3672、1事件，单侧97.5%上界4.2759%，可行性见证门槛通过。它不是学习baseline。新的`learning-screen`已实际启动两算法各256回合、批16、seed101，并在最终checkpoint各执行20开发seed双模式评估；工程和可行性硬准入均通过。原32回合pilot、所有任务诊断保持完成状态；算法实现SHA、OT合同及官方模拟器没有改变。

最新更新：两算法32回合均已完成。MAPPO-L/MACPO训练事件率为28/32、29/32；开发随机均值−5061.5、−4877.5，分别2/2事件；greedy−6373、−9097，分别2/2事件。预算5%未满足，两次更新不作为算法收敛或环境不可行的证明。五actor与双critic、checkpoint/优化器有限、重载及评估隔离检查通过。

用户随后明确终点是环境达到论文主验证任务水平，后续按[主任务验收](OT_MAIN_TASK_ACCEPTANCE_zh.md)继续。已完成的固定合法告警恢复规则四开发场景均无事件、平均−364.5，说明“低回报/高风险是任务必然”尚无依据；也不能把该规则的零事件小样本当总体可行性认证。

**工程验收已通过，两算法各32回合的有界试测已启动，顺序执行MAPPO-L与MACPO。** 每批16完整回合，每算法两次更新；训练seed101。完成后各在开发54001/54002执行随机主模式和greedy辅模式，共四回合评估。单任务上限6小时，至少30GiB磁盘余量，STOP与进程身份核验生效。

独立命名为`mappo_lagrangian-ot-finite-horizon-v1`与`macpo-ot-finite-horizon-v1`。此处“baseline测试”是新任务的适配验收、稳定性和可行性探索，不是主论文规模的正式效果实验，也不声称原版算法理论保证成立。旧可用性调参台账的两个候选不被重标为新任务实验。

## 有限时域适配

- 采用同一锁定作者网络/优化器与离散CPO求解器。actor输入仍是本agent合法Blue向量；集中reward/cost critic仍只拼接五个Blue向量，不输入隐藏OT状态。
- 新的在线monitor只向学习器发出整个回合首次超限的单次事件成本，监测完整性并在回合结束与离线分类器核对。
- 奖励和事件成本均使用完整有限回合的**未折扣Monte Carlo回报**，每个真实terminal截断。训练奖励仍按0.01缩放，日志使用未缩放官方回报。
- reward/cost advantage均用原始回报减采样时value baseline，再乘`总transition数/完整回合数`。训练器对transition取均值后得到每回合求和的局部policy-gradient surrogate；不对cost advantage单独归一化，避免将概率单位抹掉。
- MAPPO-L在一次批次的所有PPO epoch中固定lambda，最后仅按`lambda += dual_rate * (batch_event_rate - .05)`投影更新一次，不乘回合长度或`1-gamma`。
- MACPO复用作者CPO求解与回溯；仅在其`trpo_update`调用期间把唯一的约束重缩放乘数`1-gamma`置为1。局部兼容值在finally恢复，不参与任何return计算。准入程序验证锁定方法中`self.gamma`确实只出现于该缩放处，否则拒绝运行。
- 顺序更新沿用联合importance factor，并用先前agent的局部surrogate变化调整当前CPO gap。budget始终为每回合事件概率0.05，绝不换成24.95。

这是有限样本局部surrogate优化，不等于真实新策略的风险得到保证。MC有较高方差；隐藏服务状态及事件历史没有进入critic，可能影响估计质量和可学习性。使用普通Blue信息的value baseline不注入隐藏信息，但也不能凭该实现宣称完美Markov表征或精确风险critic。

## 已完成的验收

### 21项相关测试

服务采集/聚合6项、离线协议分类6项、在线事件与MC回报6项、有限时域优化适配3项，全部在正确的锁定解释器通过。包含：

- 连续16步允许、17步触发、跨使命不重复计成本、回合间清空锁存、缺失或乱序采集拒绝训练。
- 真实terminal不串回报；未折扣first-hit成本和回合标签一致。
- 两步Bernoulli完整枚举中，带每回合尺度的事件surrogate梯度与解析风险梯度一致。
- PPO epoch不重复更新dual；CPO窄作用域兼容值在异常时也恢复。

### 两算法真实集成

每算法从头训练2个完整499步回合，完成一次更新，再重载checkpoint并在两个开发场景跑随机/greedy双模式评估。共12个完整回合、5988个transition；不将其结果当作性能验证。

两个算法五个actor、reward critic、cost critic均更新且有限。MAPPO-L的实际KL约0.000625–0.001045，lambda从0.78变为约0.780475，符合一次批次更新；MACPO实际KL约0.0000285–0.0000368。两个完整checkpoint及全部优化器状态有限，重载后权重一致，评估前后权重不变。

短测训练事件率均为1，明显不满足5%目标。此结果只说明从随机初始化开始的两个训练回合未达到约束，不能证明任务不可行。没有为了通过测试放宽要求。

### 扩大试测的硬门槛

`ops/ot_continuity_baselines.py`在`bounded-pilot`模式要求已经完成的集成control，并自动检查两算法的真实更新、checkpoint/优化器有限值、同一runtime/contract SHA、事件预算、重载和双模式评估，未通过不得启动。

当前暂存代码与活动control/source独立，不覆盖服务器用户配置。运行绑定实现摘要、模拟器与依赖、上游、配置、contract SHA及runner SHA。停止发信号前核验PID+启动时间+命令；worker也会检查监督进程身份，监督消失则退出。

## 有界试测参数与解释

两算法沿用各自初始256双隐藏层和actor/critic lr5e-5。MAPPO-L PPO epoch10、lambda初值0.78、dual rate0.0005；MACPO epoch1、KL阈值0.01、fraction_coef0.1。未将旧任务中候选选参结果冒充新任务最优参数。

当前采用单环境顺序采样、8模型线程、每批16回合冻结策略；因此实际采样并行度与旧16环境任务不同，不直接比较墙钟时间。每50回合与最后保存checkpoint；当前32回合只产生最后checkpoint。不写完整回放，保留训练/更新/OT测量摘要、配置与来源。新任务使用独立run分组，当前集成2次+有界试测2次共4次，不删除旧任务关键模型。

试测完成仅标`phase_complete`，检查数值、surrogate/真实KL、权重变化、critic误差、lambda及事件率；不自动延长、不自动追加超参数候选，也不因2个开发场景结果称作满足总体5%风险。完整论文实验仍需任务可行性、算法适配审计、训练seed复核和独立风险评估证据。

## 命令入口

```bash
python ops/ot_continuity_baselines.py \
  --control NEW_CONTROL --episodes 2 --batch 2 --scope integration

python ops/ot_continuity_baselines.py \
  --control ANOTHER_NEW_CONTROL --episodes 32 --batch 16 \
  --scope bounded-pilot --preflight-control PASSED_INTEGRATION_CONTROL
```

control必须全新，不支持resume，不自动重启旧任务。数值协议初始快照中的“training未启用”保留为当时设计状态；新试测control的`admission.json`记录工程门槛通过后对有界探索的启用，不修改旧快照。
