# CAGE4 业务可用性 cost 与预算（v1）

## 结论与适用范围

采用**任务阶段加权的 Green 业务失败比例**作为共享团队 cost。默认主预算 `d=0.2474800289213457`；严格档 `0.22319008002191076`、宽松档 `0.2586557460721998`，用于三档敏感性分析。不要将百分数 24.748 直接填入 [0,1] 的 rate 字段。

这是本项目新增的约束实验协议。CAGE4 官方提供 reward 与业务优先级，没有给出此 CMDP cost 或这些预算；MACPO 论文也没有给出 CAGE4 专用阈值。预算已经在当前锁定的真实 CybORG、FiniteStateRedAgent、500 配置步上校准，不能直接移用于其他红队、horizon 或 reward 定义。

## 为什么这样定义

[CAGE4 官方说明](https://github.com/cage-challenge/cage-challenge-4)要求同时维持关键业务并防御攻击。[官方 BlueRewardMachine](https://github.com/cage-challenge/cage-challenge-4/blob/main/CybORG/Shared/BlueRewardMachine.py)按任务阶段、源主机网段区分 Local Work Failure（LWF）、Access Service Failure（ASF）与 Red Impact（RIA）。例如活跃作战区的 LWF 权重从 1 提升到 10；有些阶段 ASF 权重为 0。

据此，主 reward 继续使用完整官方 reward（包含 RIA）。cost 单独关注业务可用性，避免简单把负 reward 再当成 cost，也避免把所有 Restore/BlockTraffic 都武断叫作误报。正常防御也可能造成停机，红队同样可能导致业务失败；当前成本衡量**端到端可用性损失，不做因果归责**。要研究“仅由 Blue 造成的附带损害”，还需设计成对反事实实验，不应由本 cost 名称冒充。

[MACPO/MAPPO-L 原论文](https://arxiv.org/abs/2110.02793)以受约束 Markov game 比较 reward 和 cost。[作者代码](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation)的阈值属于 MuJoCo 任务，不适合直接搬来当 CAGE4 的预算。以下任务定义与阈值选择是我们的明确研究设计。

## 数学定义与边界情况

对于 step t 实际执行的 GreenLocalWork/GreenAccessService，按官方奖励逻辑过滤无 active session 的 agent，取首个执行 action、首个 action observation 的 success、当前 mission phase 和源主机网段。权重为官方相应 LWF/ASF 的绝对值。

令 E_t 为这些任务中权重大于 0 的任务集合：

```
W_t = sum(weight_j for j in E_t)
F_t = sum(weight_j * failed_j for j in E_t)
c_t = F_t / W_t                  if W_t > 0
c_t = 0                         if W_t == 0
J_C(episode) = sum(c_t) / T
约束：E[J_C] <= d
```

每步 c_t 在 [0,1]。零权重任务不进入分子和分母；无业务步记 cost=0，但独立保存 `no_task_steps`。同时保存 eligible/failed tasks、attempted/failed weight、inactive tasks，避免把未执行业务当成“100% 成功”。无法识别的 Green outcome 会报错，不能默认为成功。本次40个场景所有步均有正权重业务，inactive task 为0。

另报告 `request_weighted_failure_rate = sum(F_t)/sum(W_t)` 作为辅助指标，不能与 `mean_cost = mean(c_t)` 混称。前者按请求权重，后者按时间平均；CMDP 优化和预算使用后者。建议论文同时报告业务覆盖、分母与无业务步数；不能靠降低覆盖率来宣称可用性提升。

全局业务 cost 只产生一个标量，safe trainer 给五个 agent 同一团队约束。网络 Ground Truth 与 Green 详细事件用于研究者采集，不输入 Blue actor；actor 仍只接收本 agent Blue observation。MAPPO/IPPO/HAPPO 也被同样测量 cost，但不把它加入无约束算法损失。

## 不看最终测试集的预算校准

预先固定参考策略为 Sleep（全部合法 Sleep 动作），使用独立 seed：

| 用途 | seed | 数量 | 是否用于调参 |
|---|---|---:|---|
| 预算校准 | 41001–41020 | 20 | 仅估计参考分布 |
| 预算独立验证 | 42001–42020 | 20 | 仅检查参考均值可行性，不重新调预算 |
| 模型最终评估建议 | 53001–53100 | 100 | 不参与预算选择 |

对20个校准 episode 的平均 cost x_i，定义：

```
U = mean(x) + 1.729 * sample_sd(x) / sqrt(n)
d_main = min(1, max(q75(x), U, 0.001))
d_strict = min(d_main, q50(x))
d_loose = max(d_main, q90(x))
```

1.729 是 df=19 的单侧95% t 近似临界值，n>20 时保守使用。该区间依赖独立样本及均值近似，不是有限样本、无分布假设保证。主预算取 q75 与参考均值上界中的较大者，旨在初始比较中容许随机场景波动，避免设一个没有证据可达到的严格 SLA。0.001 仅防止全零校准下主预算退化；本次不生效。分位数使用 NumPy 默认线性插值。

### 当前 Mac Studio 实测结果

完整原始校准/验证摘要在 [availability-v1.json](../cage4-baselines/configs/safety/availability-v1.json)。不含机器名、私有 IP、用户名、密钥或学习模型数据。

| 项目 | 值 |
|---|---:|
| 校准 Sleep 平均成本 | 22.1998075% |
| 校准均值单侧95% t 上界 | 23.8621494% |
| 主预算 | **24.7480029%** |
| 严格预算 | 22.3190080% |
| 宽松预算 | 25.8655746% |
| 独立验证 Sleep 平均成本 | **20.3761205%** |
| 独立验证参考均值是否低于主预算 | 是 |
| 每个 episode 实际 transition 数 | 499（配置 horizon=500） |

验证只是说明已知参考策略在这20个独立场景的样本均值可行，不能保证所有 seed、所有 episode 或任何训练策略都满足预算；也不是对主预算最优性的证明。三档相近是参考分布的实测结果，不应为了好看的曲线任意拉大差距。若论文重点是更严格业务 SLA，另行预注册目标、增加校准量并验证至少有可行策略；不要根据最终测试结果事后移动阈值。

### 算法输入的单位

作者 MACPO/MAPPO-L 接收 episode **累计 cost**。因此每个完整回合以实际 T 设置：

```
cost_sum = sum(c_t)
episode_budget = d * T
```

本版本 T=499 时，主预算约为123.4925；20步 smoke 实际19步时约4.7021。不得把0.24748直接与累计几十/上百的 cost 比较。作者 trainer 使用 gamma=.99 的 cost advantage 和 `(sum_cost - sum_budget)*(1-gamma)` 约束 surrogate；这是作者实现惯例与我们有限时域率预算的桥接，不代表精确有限时域约束的理论保证。最终仍用未折扣 episode mean cost/超限率判断。安全失败率低但 reward 很差的策略不能只凭 cost 宣称防御有效。

选择敏感性档位：复制算法 YAML，设置 `budget_level: strict` 或 `loose`（默认 main），训练和评估使用同一配置。所有方法在同档位共用数据定义/阈值。不要修改 JSON 的校准结果来切换档位。

## 重新校准与复核（远程 Mac Studio 终端）

```bash
cd "$HOME/cage4-research"
PY="$HOME/cage4-research-env/bin/python"
export PYTHONPATH="$PWD/protocol:$PWD/cage4-baselines/src"
"$PY" -m cage4_baselines.safety.calibrate \
  --steps 500 --output "$HOME/cage4-experiments/availability-recalibration.json"
```

新结果先审查 `status`、验证均值、覆盖和版本，再用于新实验。不要覆盖已完成研究对应的预算文件。入口检查 reward 源码 SHA256 与校准一致、horizon一致、校准通过；模型 checkpoint 还绑定预算文件 SHA256。更换 CybORG/reward 后必须重新校准。

本次 CybORG commit 为 `8c3c50ca54b176c2de199847944e8dcc035497e3`，奖励源码 SHA256 为 `cdcdf1bf9b2f6b30e6bfc859ac85b9475ecf8ed30f7bb86f6c3101233ea3b37f`。Python3.10.22、NumPy1.26.4。CybORG/Gym/NumPy 版本变化可能改变随机序列和动作执行结果；单凭相同整数 seed 不保证跨版本完全复现。

训练记录保存版本、每步成本、Episode mean/sum、辅助请求加权失败率、预算、超限量与超限指示。Server 只读这些文件；不计算或修改策略，不执行 checkpoint。
