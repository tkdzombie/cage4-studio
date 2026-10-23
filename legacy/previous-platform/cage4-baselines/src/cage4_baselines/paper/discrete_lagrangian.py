"""Categorical entropy port for the author continuous-task MAPPO-L core."""
import torch


def masked_entropy(dist, active_masks, tpdv):
    entropy = dist.entropy()
    if active_masks is None:
        return entropy.mean()
    active = torch.as_tensor(active_masks, **tpdv).reshape(-1)
    if entropy.ndim != 1 or active.shape != entropy.shape:
        raise ValueError('Expected one entropy and activity flag per transition')
    if active.sum() <= 0:
        raise ValueError('Entropy requires at least one active transition')
    return (entropy * active).sum() / active.sum()


def actor_class():
    from mappo_lagrangian.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    class CategoricalActor(R_Actor):
        def evaluate_actions(self, obs, rnn_states, action, masks,
                             available_actions=None, active_masks=None):
            if available_actions is None:
                raise ValueError('CAGE4 MAPPO-L requires legal action masks')
            features = self.base(torch.as_tensor(obs, **self.tpdv))
            if self._use_naive_recurrent_policy or self._use_recurrent_policy:
                features, _ = self.rnn(features, torch.as_tensor(rnn_states, **self.tpdv),
                                       torch.as_tensor(masks, **self.tpdv))
            dist = self.act.action_out(features, torch.as_tensor(available_actions, **self.tpdv))
            action = torch.as_tensor(action, device=self.tpdv['device']).long()
            active = active_masks if self._use_policy_active_masks else None
            return dist.log_probs(action), masked_entropy(dist, active, self.tpdv)

    return CategoricalActor
