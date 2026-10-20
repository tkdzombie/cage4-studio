# 预算、return与训练长度

## 新实验的约束

Run/Debug `cage4-baselines/src/cage4_baselines/experiments/run_baseline.py` 的新默认预算来自 `configs/safety/availability-target-v2.json`：

| 档位 | episode平均cost上限 | 实际499次transition的累计预算 |
|---|---:|---:|
| strict | 0.03 | 14.97 |
| main（默认） | **0.05** | **24.95** |
| loose | 0.10 | 49.90 |

cost仍是[任务阶段加权的Green业务失败比例的时间平均](SAFETY_COST_zh.md)，不是感染主机数、Blue误操作概率或未加权业务请求失败率。主预算是要求期望episode平均cost不超过5%；不是每一步或每个episode必然达标。

历史Sleep参考策略20个校准场景平均cost为0.221998，新主目标要求比该参考均值降低约77.5%，比旧主预算0.247480收紧约79.8%。三档是明确的研究设计，没有来自CAGE4或MACPO的官方SLA。保留原40个参考测量、reward源码SHA和cost版本，新增schema `cage4.budget-target/1`：`status=specified`、`feasibility=not_established`。参考策略未达到新目标如实记录，不能把“指定目标”当成“已证明存在可行策略”。不根据最终测试集修改预算。

所有比较方法应使用相同预算档位、seed划分、观测、交互预算与评估动作模式。MAPPO/IPPO/HAPPO只测量cost，无约束损失不会因调预算而改变；MAPPO-L/MACPO才使用阈值优化。自定义算法需明确实现约束部分，不能仅改显示阈值就称作安全算法。敏感性分析只改YAML中的`budget_level`，不覆写已存实验的budget.json。

正在运行的旧实验继续保存自己的旧预算、源码和指标，下一次运行入口才使用新目标。用旧权重算新阈值超限率属于另行标注的事后分析，不能冒充“以新预算训练”的结果。新训练自动保存`source/cage4_baselines/`；`ops/evaluate_baseline.py`优先加载该次受信任实验的源码，仍核对checkpoint中的源码、依赖、配置、预算SHA和训练/评估seed隔离。

## ret怎样判断

本项目主要看 `official_team_return`，每步共享Blue reward只算一次，完整累计所有实际transition。越接近0越好。`team_return`把五份共享回报相加，通常是其5倍，不可混用；训练内部reward_scale=0.01也不改变日志里的原始官方回报。与官方评估脚本严格对照时还需使用排除终止transition的 `official_script_compat_return`，不能把不同终止处理混为完全相同协议。

[CAGE4官方总结论文](https://ojs.aaai.org/index.php/AAAI/article/view/35158)表2在原始CC4列报告：UC启发式约−113±35、Lancer约−118±40、Punch约−142±44、Cybermonic MARL约−193±84，均为100个500步episode的累计回报均值及标准差。它们是不同策略、wrapper与训练方案的公开参考，不是当前MAPPO的保证值；需要复核模拟器版本、动作、seed和计分口径才能作严格横向比较。

因此不能把−3000甚至−7000叫作“接近论文水平”或已经收敛。工程调试可先观察独立验证均值是否稳定改善，例如逐步接近−1000、−500，再争取数百量级；这些只是项目阶段目标，不能当作通用合格线。无约束reward改善和满足cost必须分别报告。最终按预定独立测试集、多个独立训练seed报告均值、离散程度、成本与超限率，而不是只挑一条训练曲线。

## 官方运行多少回合、多少步

[官方README](https://github.com/cage-challenge/cage-challenge-4#evaluation)规定评估100个随机episode，每个配置500 timesteps，对手FiniteStateRedAgent；这是**评估数量，不是训练数量**。官方未统一规定所有算法训练多少episode。[Cybermonic作者实现](https://github.com/cybermonic/cage-4-submission#agent-training)报告其GNN/PPO方案约50,000个训练episode后收敛，这不是本项目MLP-MAPPO的收敛承诺。

当前锁定版本配置500步，适配器实际记录499个transition。训练2000回合约998,000环境交互；50,000回合约24,950,000交互，不能再乘5个Blue agent当作环境步数。16个并行环境只改变墙钟速度，不减少总交互预算。2000回合用于初步诊断；进一步可预先统一设定10,000或50,000回合预算，再根据独立验证曲线和固定停止规则比较。不要自动因一次训练没收敛就延长某一个方法而不给其他方法同等预算。

## 2026-10-20开发验证发现

第700回合checkpoint用未参与训练的开发seed 54001–54020，确定性取最大概率动作评估：平均官方回报−6753.35，平均cost=0.198820。按旧主预算超限3/20，按新主预算事后核算超限20/20（0.15的候选预算仍超限16/20）。说明新阈值更有区分度，也说明该checkpoint的独立评估表现仍弱；不是最终测试结果或可行性证明。

同轮训练后期100个episode的采样回报约−1471、平均cost约0.0403，但训练时随机采样动作，不能直接拿来替代上述确定性验证。出现差距时应区分动作选择模式与跨场景泛化，再检查训练预算和优化参数；不能仅凭训练曲线宣布收敛。

另取同一checkpoint和相同开发seed 54001–54004，仅改变评估动作选择为随机采样，完全不更新权重：平均回报−1856.00、cost=0.050408；对应确定性评估为−6886.75、cost=0.211267。这个4场景诊断表明动作选择模式贡献了明显差距，不能仅归因于seed泛化，也不能选择性只公布更好的模式。主评估仍保持原定确定性方式，并明确报告随机策略诊断；后续论文若改用随机策略作为主协议，应在所有方法中统一预先设定、重复动作采样并固定seed。

最初考虑15%作为主预算，但该随机策略诊断说明它仍过于宽松，因此开发阶段最终指定5%主目标，3%/10%作敏感性档位，之后冻结目标，不能再追着最终测试表现移动。单个MAPPO权重的少量开发场景不构成可行性证明。
