# 写自己的算法

日常只改两处：`algorithms/my_algorithm.py`写方法，`configs/my_algorithm.json`设置实验。

1. `MyAlgorithmPolicy.loss()`目前直接使用PPO loss，未修改不算新算法。可以扩展actor/critic目标与轨迹处理；postprocess必须仅调用一次父方法，避免重复缩放reward。网络由`cage4/training.py`中model设置；完全不同的学习器可直接复用`cage4/environment.py`，不强制继承PPO。
2. 配置中填写唯一run路径，设algorithm为my_algorithm。当前模板与代表性PPO一样，从零初始化、24worker、150更新。PyCharm Run根目录run.py，默认读取该配置。
3. 固定环境协议、观测权限、奖励、动作mask和训练预算。改Shield/约束/输入时明示变化并做公平比较。初始化也要匹配；author时仅载入作者权重并将critic输出单位转换至learner scale，新建优化器，不能称精确续训。
4. 结束后看`runs/<运行名>/evaluation.json`中的means.final与PPO同预算末尾对比。正式用final，best仅诊断；不能拿训练回报最高点代替共同场景评估。

初始与末尾都用同一组场景、同一探索随机数，每回合499步，团队reward计一次。不同网络结构需适配checkpoint；旧模型的观测/动作维度保持不变。正式报告可另冻结未用于调参的评估场景，不把四个开发场景当作泛化保证。

旧两份保留模型可用：

```bash
.venv/bin/python run.py evaluate --run models/ppo-author --endpoints final
.venv/bin/python run.py evaluate --run models/ppo-scratch --endpoints final
```

新运行自动保存config、全部相关源码SHA、真实checkpoint及指标。不要在run/source副本里修改算法。维持`algorithms/ppo.py`与已有算法类的导入路径，避免旧checkpoint无法加载。
