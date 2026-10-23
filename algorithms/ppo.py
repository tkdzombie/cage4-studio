"""Verified PPO reward-unit adaptation; official environment rewards stay raw."""
from ray.rllib.algorithms.ppo import PPOTorchPolicy
from ray.rllib.policy.sample_batch import SampleBatch

class ScaledRewardPPO(PPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] * self.config.get('learner_reward_scale', 1.0)
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)
