# 修改算法与公平对比

当前环境为官方 CAGE4，主线为五个独立策略的平坦 PPO。服务器 PyCharm 打开 `/Users/tiankedi/cage4-research`，解释器选择该目录 `.venv/bin/python`。

1. 修改 `algorithms/my_algorithm.py` 的 `MyAlgorithmPolicy.loss()`。模板原样返回 baseline loss，未修改不算新算法。沿用 RLlib TorchPolicy 接口；critic/actor 共用上游 action-mask 模型。改网络时在 `train.py` 注册新模型并设置 model 配置；新形状与作者 checkpoint 不匹配时用 scratch。
2. 打开 `experiments/run_baseline.py`，设置唯一 `RUN_NAME`，保留 `ALGORITHM='my_algorithm'`；选择 `INITIALIZATION='author'` 或 `'scratch'`。右键 Run 开始训练，Stop 会尽可能保存末尾模型。不要在 source 快照中编辑。
3. 对照 PPO 用相同初始化、训练预算、超参数、训练 seed，将 `ALGORITHM='ppo'` 并换一个 RUN_NAME。当前现成 author 基线训练 30 更新/359280 env steps；scratch 基线 42 更新/502992 env steps。要比较等预算终点，应另跑匹配对照或明确报告实际预算；batch8192是请求值，完整回合采样实际11976/update（24 workers）。更改workers会改变实际batch与轨迹，不能默认为完全相同实验。
4. 用下面命令恢复完整模型并评估。同一组场景和探索随机数，使用 final，避免训练回报择优。新 run 会保存算法、环境、runner 的源码与SHA。

```bash
cd /Users/tiankedi/cage4-research
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python compare_flat.py \
  --run runs/my-algorithm-author-s321-v1 --endpoints initial,final \
  --seeds 54001,54002,54003,54004 --output my-algorithm-comparison.json
```

若修改网络结构，`compare_flat.py` 当前与作者模型的形状检查也需相应改造；当前入口支持保持相同观测/动作接口的 PPO loss/更新扩展。完全不同的学习器应复用 `primitive_env.PrimitiveEnv` 自建 runner，保留同样 reset/step、终点、mask、奖励和记录约定，不强行塞进 PPO 子类。

评估脚本默认从 run/config.json 读取环境协议，显式不同协议会拒绝。比较 JSON 放项目根目录且文件名含 comparison 看板会发现。看板只显示同协议、同 run 的结果。不同实验的数值可从各 comparison JSON 的 means.final 对比；actor/critic更新诊断在每个run的metrics.jsonl里。

必须保持：官方环境奖励（展示team reward仅计一次）、作者state-derived观测权限、动作集合与mask、500配置步/499transition、红队类型、场景分布一致。learner reward scale .01 保持不变。加Shield/额外约束/新观测时明示协议，提供公平消融；不要悄悄改变baseline环境。

当前作者参考无Block/Allow动作，不是与你的全控制策略严格同动作空间的主比较项。主对照是本项目同协议 PPO。54001–54004已用作开发诊断，正式报告应另冻结未用于调参的评估场景。

`initial/latest/best/final` 是真实 Ray checkpoint。warm-start 从权重开始，创建新优化器，非精确续训。新算法 checkpoint依赖保存时的 Python 模块，移动或改名算法类前保留兼容模块，历史 checkpoint 不随意删。

## 评估服务器保留的baseline

历史曲线/结果已清理。现成作者初始化baseline为 `models/ppo-author/final`，冷启动为 `models/ppo-scratch/final`；两者各保留config.json用于读取正确环境协议。仅评估末尾，不请求已删除的initial/latest：

```bash
.venv/bin/python compare_flat.py --run models/ppo-author --endpoints final \
  --seeds 54001,54002,54003,54004 --output ppo-author-comparison.json
```

新实验仍保存到runs/<唯一运行名>，训练后再生成自己的评估结果。
