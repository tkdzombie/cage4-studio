# 结构化观测与训练协议 v2

## 输入设计

统一编码器是 `blue-belief-slots/1`，替代旧版哈希向量。不使用随机 IP、PID、字符串哈希，也不设计专家防御策略或复杂安全评分。输入是普通 float32 向量，所有算法使用相同字段和合法动作掩码；MLP 可以直接接收，后续可按保存的 schema 还原主机矩阵接入 attention/GNN。

向量顺序为：步数/配置 horizon、5维 agent one-hot、4维动作结果（UNKNOWN/TRUE/FALSE/IN_PROGRESS）、动作类别 one-hot、4×8 消息位、固定主机槽位矩阵。主机顺序使用官方 BlueFixedActionWrapper.hosts()，与该 wrapper 的主机动作索引一致；不同 agent 补零至共同最大主机槽数。每次运行保存 `observation-schema.json` 和 `spaces.json`，不再手填 obs_dim。

每个主机槽的前 12 个基础字段：

| 字段 | 含义 |
|---|---|
| known | 本回合曾出现在该 Blue 的观测中 |
| observed_now | 本次观测包含该主机 |
| own_session | 最近可见 Sessions 是否非空；不是红队 Session |
| process_event | 本步收到无 Connections 的进程记录；初始化进程清单不算事件 |
| connection_event | 本步收到带非空 Connections 的进程记录 |
| process_history / connection_history | 本回合累计可见事件标志 |
| analysed | 成功 Analyse 已执行且结果尚未被恢复操作清除 |
| file_count | 最近成功 Analyse 返回的文件数，截断至10后除10；不推断真实感染状态 |
| alert_age / analysis_age | 距最近事件/分析的步数除 horizon，截断至1；无记录为1，结合标志区分 |
| last_target | 本次动作结果对应的目标主机 |

成功 Restore 清除该目标的事件及分析记忆；Remove 只记录清理尝试，再接收同一步新事件；这表示本地记忆重置，不保证主机已清除攻击者。每次 reset 清空全部历史。未知主机和 padding 不冒充已观察的正常主机。

编码器只接受该 agent 的实际 observation、自己的公开固定动作槽和本地时钟。它无法访问 environment/controller/state、其他 Blue 私有观测、真实攻击状态、红队 Session 或可视化回放。集中 critic 拼接五个 Blue 向量；safe learner 额外接收团队 cost 标量，actor 不接收 cost 明细。IN_PROGRESS 时仅允许 Sleep，避免将被环境忽略的动作选择用于策略更新；其他时候不增加专家式动作屏蔽。

## 依据与差异

固定主机槽位及观测历史参考 [CAGE4 官方固定动作接口](https://github.com/cage-challenge/cage-challenge-4) 和 [H-MARL 论文](https://arxiv.org/abs/2410.17351)、[作者实现](https://github.com/adityavs14/Hierarchical-MARL/tree/6fe960f5931f2d7dc5ef522f09ca4a179f377ee4/marl-1policy)。审查文件为 BlueFlatWrapper_CC4.py。这里只借鉴结构设计，没有复制作者代码或接入 H-MARL 算法。

作者 wrapper 还从 state 读取进程/连接事件、流量阻断及通信策略，并使用文件名称和诱饵构造专门 IOC。本工程为保持 Blue 信息边界及通用 AI 输入，省略内部状态、专家 IOC 和自定义通信协议，只编码 Blue 真正收到的字段。因此这是**明确规定的统一比较协议**，不是 H-MARL 观测的逐项复现，也不是某个唯一的“官方 AI 标准”。

## 正式训练设置

五个策略分别训练、不共享权重，CPU；actor 和 critic 各两层256隐藏单元（上游先建第一层，layer_N=1再加一层）。lr/critic_lr=5e-5，gamma=.99，GAE=.95，PPO clip=.2，entropy=.001，PPO每批10轮。MACPO继续使用作者约束更新与单次 trust-region 更新，不使用PPO多轮替代。

默认每16个完整回合组成一个批次，500配置步在锁定环境实际499步，通常每批7984条团队transition。GAE在每个回合终止处截断；集中输入和顺序importance factor保持一致。安全算法使用批次内平均episode cost，与 rate×平均episode长度比较，不把8回合的累计cost误当单回合。最后不足16个回合时消费真实剩余数据。每回合写回报/cost，只在更新发生时写loss，空loss不是训练故障。

此批次规模适合现有单机训练，并非论文中的百万样本/30 worker设置。默认2000回合约99.8万次环境transition，作为统一起始预算；它不证明收敛。独立训练seed、评估集合、回报/cost口径和算法来源见 [PAPER_BASELINES.md](PAPER_BASELINES.md)。应先按共同预算比较，再依据独立验证曲线统一延长，不能只给表现差的方法额外数据。

旧哈希编码checkpoint无法加载为新结构化模型。新训练必须从头开始，旧记录留在最近5次范围内时仍可只读查看，但不应与v2结果混合成同一学习曲线。

## 数据保留

实验按开始时间只保留最近 **5次运行**，一次运行可以包含数千个episode。第6次及更早运行的指标、回放、权重和TensorBoard文件会直接删除，不归档、不放废纸篓。每次创建及结束运行都会执行清理；网页默认只展示最新运行。

仍在运行或无法可靠核实进程身份的写入者受保护，必要时临时超过5次，结束后再清理。删除操作带目录锁，不跟随符号链接，只管理可识别的实验目录。测试和验收必须使用临时实验目录，结束后清除，避免挤掉用户最近5次实验。此保留策略意味着启动较多新实验会删除早期论文结果；需要长期论文存档时应先明确另行改变保留策略。

## Belief 与 staleness 的最终接口

最终编码版本为 **blue-belief-slots/1**，上文12维是其基础字段，现增加独立通道的掩码、年龄、新鲜度与动作记忆。名称中的 belief 是确定性观测记忆，不是贝叶斯后验。Remove成功只记录动作结果；因为官方Remove会保留特权会话、且不删除文件，**不清空事件或文件记忆**。Restore成功才重置当前告警记忆，同一步新告警仍可重新置位。

位置：raw observation 由 CybORG 的 Observation/Monitor/Analyse 等动作生成；官方 BlueFixedActionWrapper 原样转交 Blue 字典并提供固定动作槽。本工程的 `envs/adapter.py` 在其后调用独立 `envs/observations.py`；`envs/belief.py` 实现记忆时间、衰减与观测扰动。主机槽来自官方hosts()，subnet来自公开槽名称，不从隐藏state提取链路或阻断状态；动作结果来自自身observation的action/success。

锁定版本实测：5个Blue的MLP输入均为 **2345**维，合法动作掩码 **242**维。前50维为全局头，后51×45为主机矩阵；agent0–3各17个可能主机槽，agent4为51个。全局头索引：0时间；1–5 agent ID；6–9 success；10–17动作类别；18–49消息位。8种动作类别依序为AllowTrafficZone、Analyse、BlockTrafficZone、DeployDecoy、Monitor、Remove、Restore、Sleep。

每个主机槽内部索引如下，总向量位置是 `50 + 45×slot + field`：

| 索引 | 字段 |
|---|---|
| 0–11 | 上表基础12字段，顺序不变；Remove不再清除历史 |
| 12–14 | server/user/other one-hot |
| 15–18 | process/connection/file/host的本源采样时刻掩码 |
| 19–22 | 同四通道的线性年龄 |
| 23–26 | 同四通道的对数年龄 |
| 27–30 | 同四通道的确定性新鲜度 |
| 31–38 | 该主机最近一次有结果的动作类别one-hot，顺序同全局头 |
| 39–41 | 动作年龄、动作是否成功、是否存在动作结果 |
| 42–44 | 曾成功Analyse/Restore/Remove |

尚无该通道观测时：mask=0、age=1、confidence=0，并结合known标志使用。只收到主机信息不会刷新process/network/file时间；这些事件是正告警通道，不将“无告警”解释为完整的阴性检测。成功Analyse才刷新文件查询时间，允许返回空结果。延迟消息使用源采样时刻计算年龄，`observed_now`表示本次收到主机payload，15–18表示采样本源是否属于当前时刻，两者在delay实验中不同。

令最近可用信息源时间为τᵢₖ，Δᵢₖ=t−τᵢₖ。线性年龄为min(Δ,A)/A，对数年龄为log(1+min(Δ,A))/log(1+A)，新鲜度为exp(−λΔ)。未知信息的新鲜度为0。λ默认.01，A默认500；它衡量陈旧程度，**不是信息正确率，也不是主机安全概率**。未收到新观测则保留记忆；reset重新初始化所有通道与随机流。

### 配置消融和扰动

各算法YAML的observation段完全独立于算法，使用同一字段维度：

| mode | 事件记忆 | 年龄 | 新鲜度 |
|---|---|---|---|
| current | 仅当前收到的事件 | 关闭 | 关闭 |
| persistent | 保留历史事件 | 关闭 | 关闭 |
| age | 保留历史事件 | 开启 | 关闭 |
| belief（默认） | 保留历史事件 | 开启 | 开启 |

这是本工程同一信息权限下的4种消融；current不是官方BlueFlatWrapper。所有模式保留相同的动作记忆和公开主机槽，比较事件记忆/时龄增益。不要把它们称为已完成的GNN消融。`dropout`按主机payload丢弃，`delay`为整数步延迟，`noise`翻转收到的进程记录对应的进程/连接事件位；均作用于深拷贝，环境、奖励和动作mask不改。各agent随机流按episode seed、agent ID和配置seed派生。可分别测试dropout=[0,.1,.2,.3,.4]、delay=[0,1,3,5,10]、noise=[0,.05,.1]；初期采用单因素测试。

### 图接口与约束上下文

`PolicyInput.graph`提供纯NumPy的node_features[54,45]、edge_index[2,E]、edge_attr[E,1]、node_type[54]、node_mask[54]、global_features[50]和action_mask[242]；可用torch.as_tensor转换为PyG输入，不依赖PyG安装。51个Host槽后为3个Subnet槽；Host↔Subnet双向归属边E在agent0–3为34，在agent4为102。Subnet行前5列为可见主机占比、本步告警占比、记忆告警占比、平均年龄、平均新鲜度，其余零。mask指**可能槽位**，不是隐藏的物理存在真值。具体可见性用known字段。

这里没有未知Subnet↔Subnet链路、动态block或通信策略边，也没有捏造importance/mission_phase。当前可见API没有提供足够观测，不能用researcher拓扑补齐。现有五算法使用2345维向量和作者MLP；图接口供自定义算法使用，**没有训练GAT/GraphSAGE，不声称已做图编码性能比较**。如果要比较图编码器，需要给全部方法同样的编码器、信息和调参预算，不能只给新算法额外表达能力。

`decision_context`单独记录累计cost、最近cost、名义episode预算和clip(1−累计cost/预算,−1,1)，不混入actor向量/graph。它在metrics和回放中可读，safe训练器另收共同cost。默认预算来自researcher才能完整计算的Green业务事件；把它传给Blue actor会改变信息假设，因此当前没有启用“budget-actor”消融。若后续研究明确把团队cost当在线公共反馈，必须对所有算法同时启用并单独命名新协议。

每回合日志包含mean_age、age_p50/p95的时间平均、mean_confidence及stale_state_action_rate。后者定义为Remove/Restore动作中，目标的process/connection/file最低新鲜度低于.5的比例；零个此类动作时返回0，并同时保留分子/分母。它只是研究诊断，不参与奖励/动作屏蔽。保留budget_usage、超限布尔值和跨评估seed的violation_fraction；环境没有统一二值task-success标签，因此不编造task success rate。

### 论文表格建议与边界

主实验使用同一belief输入，对IPPO/MAPPO/HAPPO/MAPPO-L/MACPO及新方法报告跨训练seed的return、mean cost、violation fraction和交互样本数。观测消融固定一个安全算法，仅改current/persistent/age/belief；鲁棒性表固定算法与训练预算，分别改变dropout/delay/noise。图编码器和budget-actor应在真实实现并统一信息条件后再加入，不预填结果。

本实现没有完整环境状态后验、没有安全概率校准、没有全网真实图、没有新的GNN架构。它能检验记忆和信息陈旧是否有用；这些收益仍须多seed实验验证，不能靠结构设计或短程验收宣称。
