# 日常研究与扩展

在远程 PyCharm 中修改 `cage4-baselines/src/cage4_baselines` 和 configs；Server 不导入算法，Client 不参与训练。配置 schema 在 config.py；未知字段和范围错误直接拒绝。避免在网页边调参数边认为已启动的实验发生变化，每个 run 保存其实际配置。

一个完整流程：复制 YAML → 修改 seed/episodes/steps → train → 找 checkpoint → 独立 evaluate → 用相同 seeds 跑 Random/Sleep → 在网页对比。以多训练 seed × 独立评估 seed 做科研结论；交付中的两轮/三 seed 只验证代码可运行，不是收敛或优越性证据。

Actor-Critic 为五个独立网络，每个 128×128 隐藏层，输出 logits 和 value。使用动作掩码、Monte Carlo discounted returns、REINFORCE actor loss、0.5 MSE critic loss、entropy bonus、梯度裁剪和 Adam。截断 horizon 当作回合结束，没有 bootstrap；这在短 horizon 会产生偏差。选 CPU 先验证；MPS、长时间训练和内存上界需单独测。不能直接复用旧 BlueFlat 编码的 checkpoint。

train 的 `--runs` 创建唯一实验目录；checkpoint_every 和最终 Episode 写 checkpoint。SIGINT/SIGTERM 写 stopped，未完成 Episode 不进行参数更新。停止不是续训功能：本版没有 resume，不能将 evaluate 的 --checkpoint 误作恢复训练。保留完成的 checkpoint 后可独立评估。

独立评估只允许 Actor-Critic 搭配已保存 checkpoint，检查 encoder、algorithm 和维度。只加载可信本地权重。修改 obs_dim 或动作空间会被拒绝。跨不同 steps/red 的评估虽可运行，但必须在论文或报告中标明配置差异；不要把训练和评估曲线合并解释。

## 添加算法

例如新建可导入 Python 包 `my_cage4_policy`，在 YAML `plugins: [my_cage4_policy]` 显式启用；将包安装到研究解释器，或添加到研究项目的源路径。插件只由研究进程加载，不由网页导入。配置是可信代码入口，不能接受未审查的网上模块名称。

```python
from cage4_baselines.algorithms import register

@register('my_policy')
class MyPolicy:
    def __init__(self, inputs, config):
        # inputs: agent -> PolicyInput(observation vector, bool mask, labels)
        # 初始化模型与独立 RNG；勿读取实验记录、全局状态或红队遥测。
        pass

    def act(self, inputs, training=False):
        # 真实实现应按 mask 选择动作，返回 agent -> Python int。
        raise NotImplementedError

    def observe(self, rewards):
        # act 后接收本步的 agent -> float reward。
        pass

    def update(self):
        # 每个完整 Episode 调用一次；返回 agent -> loss 分项。
        return {}

    def checkpoint(self):
        return {}  # 可由 torch weights_only=True 安全读取的张量和基本容器

    def load(self, data):
        pass
```

这是接口说明，不是已实现的算法。将 YAML algorithm 改成注册名称。需要自定义超参数时显式扩展 config.py 的字段、范围检查和配置文档，避免任意未知键被悄悄忽略。需要 rollout/minibatch/跨 agent 集中 critic 时，扩展算法/runner 契约并提高协议版本或加入向后兼容字段。IPPO/MAPPO 要实现 PPO clipping、优势估计、batch 更新等真实内容；注册名称本身不算实现。

## 回放、对比、数据维护

frame 的每步动作/观测真实保存，拓扑按间隔保存，不重建未记录的中间状态。Red Session 信息为研究者 Ground Truth；合法性与成功区分见 PROTOCOL。旧 runs 的 metrics/快照只读兼容显示，旧 Actor-Critic loss 只有总值，无法凭空补齐 actor/critic/entropy。

观测台每 5 秒刷新当前指标，最多载入最近 10,000 条指标，较大实验的完整统计请离线分析。非常长的回放文件索引目前仍全量枚举，百万步规模应增加分块索引；当前不承诺大集群或高并发可视化。磁盘配额/保留策略由研究者管理，不自动删除历史。网页不自动扫描源码外任意路径。

研究记录、checkpoint、私钥、.env 不进入 Git 或发布包。源文件 Git 忽略 runs/、*.pt、日志和密钥。发布前仍须人工核对 staged diff，不能只依靠扩展名忽略所有秘密。
