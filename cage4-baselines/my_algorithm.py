"""自己的算法写在这里。

这是可运行的 MAPPO 扩展示例，默认完全沿用 MAPPO，不是新的论文算法。
使用：run_baseline.py 顶部设置 ALGORITHM = "my_algorithm"，右键运行。
不修改 cage4-upstreams 中锁定的作者代码。
"""
import hashlib
from pathlib import Path
from cage4_baselines.paper.backend import AuthorBaseline


class MyAlgorithm(AuthorBaseline):
    def __init__(self, inputs, config, upstreams, budget_rate):
        # 默认基于 MAPPO 初始化五个 actor/critic；可在这里定制网络/训练器。
        base_config = dict(config, algorithm="mappo")
        super().__init__(inputs, base_config, upstreams, budget_rate)
        self.provenance = dict(self.provenance, extension="my_algorithm",
            extension_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

    def update(self, reward_scale):
        """在这里实现自己的训练更新；目前调用原 MAPPO 更新以保证示例可运行。

        self.history 是本回合真实 transition 的列表，每项：
          (records, rewards, cost)
        records[agent] 包含 obs、cent、available、value、action、logp。
        rewards 是五个 agent 的原始 reward；cost 是共同团队标量。
        必须返回 {agent: {指标名称: 有限数值或bool}}，并清空已消费的 history。

        只修改损失/优化步骤时，也可在 __init__ 中替换 self.trainers 中
        相应 trainer；网络需保留与 act/checkpoint/load 约定一致的接口。
        """
        return super().update(reward_scale)
