"""Your algorithm starts here. Unmodified, this is identical to baseline PPO.

loss(): differentiable actor/critic objective; do not detach the final loss.
postprocess_trajectory(): advantages/targets; call super exactly once to retain
reward-unit adaptation. The environment returns raw official rewards.
For a different network, register a custom model and configure it in train.py.
For a non-PPO learner, reuse PrimitiveEnv and write a separate training loop.
"""
from algorithms.ppo import ScaledRewardPPO

class MyAlgorithmPolicy(ScaledRewardPPO):
    def loss(self, model, dist_class, train_batch):
        # Replace or extend this objective with your own method.
        return super().loss(model, dist_class, train_batch)
