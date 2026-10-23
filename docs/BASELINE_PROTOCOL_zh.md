# CAGE4 可继续研究的基线

本次按照“搜集顶会论文”的计划恢复CAGE4，保留H-MARL支线诊断，并建立同作者 `marl-1policy` 的可训练平坦PPO基线。主实验全部使用单训练seed321；没有恢复CAGE3训练、旧实验队列或创建定时任务。

主要结果、选择的模型和曲线见 **RESULTS.md**；完整错误与失败实验记录见 **AUDIT.md**；参考计划来源见 **REFERENCE_PLAN.md**。

## 当前实现

- `primitive_env.py`：官方CAGE4 + 作者平坦包装器的独立修复层。
- `train.py`：五个独立PPO，保存初始/最佳/末尾及每5轮检查点、源码和实际配置。
- `compare_flat.py`：实际恢复Policy，在共同场景和探索随机数下比较完整回合；记录动作、官方奖励分解、模型摘要。
- `verify_flat.py`：一条499步真实轨迹检查已知的mask/单位/终点/信息接口风险。
- `dashboard.py`：只读实时曲线与同协议对照，不控制实验、不调度训练。
- `hmarl_env.py`、`compare.py`：保留的层次策略支线；其弱提升与退化结果没有隐藏。

主线协议 `flat-full-actions-history-correct` 显式允许Block/Allow；保留作者的mission通信mask和Ready时不选Monitor/Sleep的先验。busy时仅Sleep，空mask保底Sleep。这是包含流量控制的原始防御动作基线，不冒称完全无先验的官方动作全集。环境奖励、攻击者FiniteStateRedAgent、网络分布、500配置步/499真实transition都未改。

修复包括：官方mask不再被原地改写；不放开不存在主机槽；多步动作保留当前可用特征；Remove不再冒充清除文件/特权感染而重置IOC；有限任务结束不bootstrap；标准RLlib环境空间适配。actor不调用红队感染/隐藏文件真值辅助函数。

## 环境与启动

远端工作目录 `/Users/tiankedi/cage4-research`，解释器 `.venv/bin/python` 链接到 `.venv-independent`，基于 `/Users/tiankedi/cage4-runtime-independent`。核心依赖不再依赖CAGE3目录。上游为 https://github.com/cage-challenge/cage-challenge-4 （8c3c50ca54b176c2de199847944e8dcc035497e3）与 https://github.com/adityavs14/Hierarchical-MARL （6fe960f5931f2d7dc5ef522f09ca4a179f377ee4）。准确版本/路径/Git状态在 `environment-independent.json`，完整版本在runtime-freeze.txt。CybORG与Hierarchical-MARL两个上游tracked源码保持不变。

```bash
cd /Users/tiankedi/cage4-research
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u train.py \
  --level flat --workers 24 --batch 8192 --iterations 30 --seed 321 \
  --reward-scale .01 --protocol flat-full-actions-history-correct \
  --gae-lambda 1 --lr .00005 --epochs 30 --minibatch 2048 \
  --warm-start Hierarchical-MARL/marl-1policy/models/train_marl/iter_199 \
  --warm-start-value-scale .01 \
  --run runs/flat-full-actions-author-init-s321
```

已有run目录拒绝覆盖；这是已执行命令，不要重复启动同名实验。所有actor/critic继续优化，没有冻结子策略。作者原critic用原始reward单位，迁移只将最终value线性层乘.01；actor原样保留。环境及展示回报始终原始单位，learner的GAE/critic使用.01单位，不增加奖励项。

使用当前已学checkpoint另起训练时，`--warm-start-value-scale`必须为1（默认），避免重复缩放。warm-start会新建优化器；不要把它称为精确恢复全部采样状态。实际运行的源码快照保存在run的source目录，并附SHA。

## 评估与动作对齐

```bash
cd /Users/tiankedi/cage4-research
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -u compare_flat.py \
  --protocol flat-full-actions-history-correct \
  --run runs/flat-full-actions-author-init-s321 \
  --endpoints initial,final --seeds 54001,54002,54003,54004 \
  --output flat-author-endpoint-comparison.json
```

每回合重置探索随机数为scenario+710000；因此模型加载顺序不会改变对照。作者参考保留其原始“不使用流量控制”策略动作集合，在同一修复仿真下评估；不把其未训练的控制动作迁移表现当作弱参考。

Blue0–3各76维观测/82动作槽，HQ为226维/242槽。run的spaces.json记录实际主机顺序、子网顺序、向量字段与示例动作标签。物理存在和合法mask每回合变化，后续grounder应通过环境元数据产生动作编号，不能让LLM猜编号。

## 证据边界

作者编码器从state读取事件、block、decoy元数据；没有新增红队session/感染真值，但也不冒称纯Blue observation协议。未来纯观测版应另命名并重新比较。

强基线使用公开预训练权重，预训练能力不计作本次学习；另有完全随机初始化的冷启动曲线证明学习链路。不同初始化与协议的曲线分别记录。四个小场景用于快速起终点诊断，不等于多训练seed统计或大范围泛化结论；后期经验平台也不等于数学收敛或论文性能复现。

模型包保留final目录、source目录、配置/空间/依赖与原始指标。在原CAGE4环境中，可执行包内source/compare_flat.py并将--run指向解压目录、--endpoints final；CAGE4_ROOT应指向含两个上游仓库的父目录。评估默认读取保存的protocol，显式不匹配会拒绝，避免误用无流量控制动作集。
