# CAGE4 H-MARL 首轮诊断

源码与配置参见 REFERENCE_PLAN.md。保留两上游 tracked 源码，不在作者目录打补丁；适配单独放 hmarl_env.py。

## 实际确认

- 官方配置 steps=500，scenario.determine_done 在 step_count>=499 结束；评估严格执行499 transition，不偷换短任务。
- 同一步五名 Blue 都收到同一团队 reward。训练每个策略各用此 reward；看板/评估只报告一份团队回报，不能五倍求和。
- 原作者 master.step 在构造 obs 以后才更新 high_actionmask。初始54001/54002两个完整回合，返回 mask 与当前 mask 不一致1116/1139次。修复当前返回 mask；相同作者策略回报仍为−1376/−900，错位变0。错位主要包括多步动作期间的无效高层选择。
- 官方 fixed wrapper 同时设置 terminated/truncated，作者 MAE 又设 __all__ terminated=False、truncated=done，产生不一致。适配明确采用有限任务：结束 terminated=True、truncated=False，不对任务终点 bootstrap。
- 原作者 callable observation_space/action_space 会使 Ray worker 返回 bound method 并尝试序列化包含 Policy 锁的环境，导致 RLock 错误。用标准 MultiAgentEnv 组合适配，不覆盖作者 callable 接口。
- 原作者 h4 train_subpolicies 会覆盖传入 investigate/recover 动作，只训练 trafficControl；不可把它称为全部子策略从零训练。当前实验为作者冻结子策略 + 新 Master 从头 PPO，这是 H-MARL Meta 的上层训练阶段。

## 信息边界

当前标识 author-state-v1，保持作者的特征协议：事件、阻断、任务阶段、decoy元数据和已接收文件/消息。源码虽然定义 _get_infected 和 _get_files_from_state，但没有调用；不可把这些未使用函数当作 actor 已读取红队感染真值的证据。也不能把作者 state-derived 接口称作纯 Blue observation。

reward、攻击者、网络生成、回合长度均未改。修复动作 mask 与终止/bootstrap 属于训练协议变更，不能混合原始/修复学习曲线。

## 当前训练协议

seed321，一个独立训练；五个独立新 Master PPO，15个作者子策略冻结；原始reward；256×256网络，lr3e−4、gamma.99、GAE.95、entropy.001、每批5轮SGD、minibatch1024、请求8192环境步，8个同步采样worker，完整回合实际每轮11976环境步。critic loss clip上限10000避免默认10在原始负reward上过早饱和；保存原始loss/EV/KL。

保留initial、best、每5轮latest与final。15轮后依据最近三个5轮窗口判断横盘候选或持续退化；这是停止诊断条件，不是收敛认证。实际接受仍需初始/已学/作者策略在相同场景的收益、动作和关键语义证据。无训练多seed，评估两个场景仅用于快速诊断，不作统计泛化结论。

## 第一轮已停止，未成功

原量纲训练在7轮停止，83,832仿真步；训练回报从−1395.7持续降至约−1685.8。第5轮checkpoint的同场景回报−1856/−3270，起点为−1685/−1666，确认并非单纯训练场景噪声。该实验保留为失败证据，不能称收敛。

下一轮只改变训练 reward 线性单位：ScaledRewardPPO 在 GAE 前将 learner reward 乘0.01。环境和原始team return不改；这不增加任何奖励项，也不改变奖励排序；critic预测相应单位，评估仍用原始单位。每轮日志另改为本轮episode均值，避免默认100episode滑窗掩盖诊断。明确比较其价值拟合EV、KL和同场景回报，不以loss变小本身当成提升。

## 底层动作与忙碌状态修复

尺度对照第5轮实际回报为−2179/−2316，依旧低于初始策略，因此尺度修正只能证明 critic 拟合改善，不能称学习成功。底层动作日志发现 [Invalid] 219/204 次：恢复子策略的 mask 从类别索引重建，重新放开了不存在主机槽；原始编码器还通过返回的共享list原地改写官方action mask。

`valid-actions` 协议：官方mask只读副本；investigate/recover/traffic三个mask与官方合法槽相交；没有可执行动作时唯一保底Sleep。不额外屏蔽攻击相关动作，不加入专家目标选择。完整499步检查通过，三个mask始终非空且不含Invalid/Padding。作者策略同场景回报−1269/−1341；这属于不同协议，不能据此宣称学习提升。

`valid-actions-info` 协议另保留多步动作期间已有的真实特征，不再全部填1：该阶段actor仍仅有固定高层分支，全部子策略仍只允许Sleep；改善的是critic看到的任务阶段/事件历史，不增加任何信息权限或动作。原作者填1会把处于不同任务阶段和风险下的忙碌状态压成同一向量，价值拟合因此存在状态混叠。

下一轮恢复从作者checkpoint核实的 PPO 主参数：lr5e−5、gamma.99、lambda1.0、30轮SGD、256×256 tanh、entropy.001；lambda1.0减少额外GAE截断对长期防御收益的影响。采样仍是短预算完整回合，minibatch2048适配小批次；reward scale.01、grad clip.5、value loss clip10000为明确的稳定性适配。24worker提高采样并行，仍单训练seed，不做多seed资格验证。每次新运行保存源代码快照和SHA，不能混合协议曲线。

## 首次实质学习证据（仍不是最终平台）

`endpoint-comparison.json`使用相同场景seed54001/54002、每回合重置共同探索随机数（scenario+710000），模型加载不影响随机流。valid-actions-info协议：

| Master | 场景54001 | 场景54002 | 均值 |
|---|---:|---:|---:|
| 随机初始化 | −1660 | −1905 | −1782.5 |
| 作者权重 | −1234 | −1541 | −1387.5 |
| 第15轮 | −1186 | −1039 | −1112.5 |
| 第18轮 | −391 | −1206 | −798.5 |

两个场景均比初始化改善，但后期检查点间仍变化；不能凭此认定稳定平台，也不把它称论文性能复现。训练18轮/215,568环境步，五个critic EV约.28–.50，参数KL正常；训练均值噪声大，约−1000附近波动。在未有这些固定对照前，已按用户要求停止训练，没有盲目跑满60轮。

`choice-diagnostic.json`表明有可学习空间：固定Investigate明显差，固定Recover均值−935.5，使用同样现有IOC的规则均值−769.5。规则只用于诊断，不加入PPO训练奖励/动作mask/示范。

## Remove / Restore 记忆语义

官方Remove尝试终止可见可疑进程，但明确不清特权会话、也不删除文件；即使没有清除感染，它仍可返回TRUE。作者erase_history_if_recovered对成功Remove和Restore都清掉文件及告警记忆，这是错误的“已安全”推断。

`history-correct`新协议对Remove不清空已有证据；只有成功Restore才清空，并仍接收同一步新事件。实际环境/奖励不改，没有引入红队真值。包装器方法回归对照：同一文件/进程证据在Remove TRUE后保留，Restore TRUE后清理；同时继续执行499步真实轨迹的mask/reward/终点检查。

后续从第18轮已学actor/critic权重在history-correct协议下warm-start；优化器重新建立，采样worker显式同步权重。它是协议切换后的适配，不冒称无缝恢复、不拼接学习曲线。新run保存初始迁移策略，单独判断后续收益/稳定性；还与旧run的随机初始化权重在同一新协议下对照。不能将修复记忆产生的收益归于PPO。

## H-MARL 平台的最终判断与转向

正确记忆协议下15轮的三个5轮训练均值为−1049.1/−1076.3/−1057.7，确实横盘。第5轮共同随机流对照退化至−2092/−1786，因此停止并保留初始迁移权重；实际保存的第15轮仍做完整对照，不能根据一个中途坏检查点预判最终表现。

最终4个场景54001–54004，正确协议：随机Master均值−922.75，迁移起点−831.0，15轮末尾−865.5，作者Master−1051.0。末尾比随机Master约6.2%改善，但逐场景不一致，两个新增场景的迁移起点甚至略差于随机Master。该证据不足以把Master训练称为可信收敛；更不能只报告旧协议下−1782.5→−798.5的漂亮结果。

因此保留H-MARL修复/权重/全部失败记录，转入同一作者仓库 `marl-1policy` 的直接动作PPO对照。`primitive_env.py`使用该目录原始EnterpriseMAE/BlueFlatWrapper/TorchActionMaskModel，修复mask副本、合法槽、busy特征、Remove记忆及有限任务终止；五个独立PPO均从随机权重开始，没有固定/预训练子策略。动作是实际Analyse/Restore等索引，环境reward、攻击者和完整任务均不改。原作者ready状态禁止Monitor/Sleep的选择规则保留，不新增针对攻击状态的专家屏蔽。

该作者平坦输入/消息编码与H4层次版存在结构差异，另命名flat-history-correct，不把两条曲线连起来。单seed321、24worker、请求8192环境步（完整回合实际11976）、256×256、lr3e−4、lambda1、gamma.99、10轮SGD、minibatch2048、learner scale.01。它是公开作者PPO的短预算训练profile，不冒称作者百万步批次/200轮配置的逐项性能复现。

## 正式直接动作主线：包含流量控制

作者 `marl-1policy` 默认USE_BLOCK=False，这不是官方环境禁用Block/Allow，而是作者的策略动作子集先验。无流量控制诊断有约−744→−631的短期改善，随后回升；它已停止，不把一个好点当平台，也不作为后续Block规则研究的主任务。

正式协议 `flat-full-actions-history-correct` 显式设置作者包装器USE_BLOCK=True，允许Block/Allow。既有mission通信策略mask保留，Ready时Monitor/Sleep沿用作者的禁止规则（Monitor本来每步自动执行；busy/空mask保底Sleep仍可用）。因此“full-actions”在这里指包含流量控制的原始防御动作，而不是宣称完全无先验的官方动作全集。奖励、攻击者、网络分布、完整回合长度、观测权限不改；从头训练另起曲线，不沿用无控制诊断的权重。

两种平坦模式的空间为Blue0–3各76维观测/82动作槽，HQ为226维/242槽。全动作随机轨迹实测同时包含77次Block与80次Allow，499步回报−4872；证明控制能力确实启用。原作者无控制checkpoint的参考评估，仍对其actor禁用Block/Allow，保留其原始动作子集，不拿它未训练过的控制动作迁移表现当作弱参考。

正式主线五个策略全随机初始化；前18轮/215,568环境步训练均值约−4647→−1483，已有清晰学习趋势。仍等待平台及实际相同场景检查点对照，不预填最终结果。

## 冷启动学习平台与强参考

正式含控制动作的冷启动共42轮/502,992环境步后停止；训练初始−4647，末轮−952.9，后期约−1000至−1300波动。相同54001/54002与共同探索随机数：初始化−4492.0，中途（第25轮）−1288.5，后期（第40轮）−884.0，末尾（第42轮）−1089.0。该结果是实质学习与后期波动平台，不能只用最佳−786点作最终性能。

作者预训练平坦策略保留其原本不使用流量控制的动作子集，在相同修复仿真中均值−183.5，明显强于短预算冷启动；因此冷启动不能称论文性能复现或强参考已追平。它仍证明从随机原始动作策略能够有效训练，学习链路没有靠预训练子策略掩盖。

为建立强基线，下一段直接从作者marl-1policy/iter_199权重初始化，在同一个含控制动作协议上继续PPO，所有actor/critic继续优化，独立记录曲线。作者critic原用原始reward单位，迁移时只将critic最终线性输出层权重/bias乘.01，匹配当前learner scale；actor原样保留。优化器新建并明确记录，不能与冷启动曲线拼接，也不能把作者既有能力记为本次训练学到。

## 独立运行时

最初复用CAGE3 runtime启动以快速诊断。已克隆依赖为 `/Users/tiankedi/cage4-runtime-independent`，新 `.venv-independent` 使用同版本Ray2.10/PyArrow14/CybORG源；`.venv` 切为该环境的链接，旧环境保留 `.venv-shared-legacy` 供回滚。当前Python、NumPy、torch、Ray、PyArrow、gymnasium及CybORG实际路径不依赖CAGE3，版本和Git状态记录在environment-independent.json。CAGE3看板/停机记录不变，未恢复训练。
