"""Explicit categorical/numerical port of the author MACPO trust-region core.

No continuous Gaussian KL or NaN categorical mean/stddev is used. Feedforward
CPU policies only. Original constrained step and line search remain upstream.
"""
import torch
from torch.distributions import Categorical, kl_divergence


def distribution(actor, obs, available_actions):
    obs = torch.as_tensor(obs, **actor.tpdv)
    mask = torch.as_tensor(available_actions, **actor.tpdv)
    return actor.act.action_out(actor.base(obs), mask)


def categorical_kl(new_actor, old_actor, obs, available_actions):
    new = distribution(new_actor, obs, available_actions)
    old = distribution(old_actor, obs, available_actions)
    # Detach ONLY the old distribution; new logits retain a second-order graph.
    old = Categorical(logits=old.logits.detach())
    return kl_divergence(old, Categorical(logits=new.logits)).unsqueeze(-1)


def conjugate_gradient(product, b, nsteps=10, residual_tol=1e-10):
    x = torch.zeros_like(b)
    r = b.clone(); p = r.clone(); rr = torch.dot(r, r)
    if not torch.isfinite(rr):
        raise FloatingPointError('Nonfinite MACPO gradient')
    if rr <= residual_tol:
        return x
    for _ in range(nsteps):
        ap = product(p)
        denom = torch.dot(p, ap)
        if not torch.isfinite(denom) or denom <= 0:
            raise FloatingPointError('MACPO Fisher product is not positive definite')
        alpha = rr/denom
        x = x + alpha*p
        r = r - alpha*ap
        next_rr = torch.dot(r, r)
        if next_rr <= residual_tol:
            break
        p = r + (next_rr/rr)*p
        rr = next_rr
    return x


def classes():
    from macpo.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor
    from macpo.algorithms.r_mappo.r_macpo import R_MACTRPO_CPO

    class CategoricalActor(R_Actor):
        def evaluate_actions(self, obs, rnn_states, action, masks,
                             available_actions=None, active_masks=None):
            if available_actions is None:
                raise ValueError('CAGE4 MACPO requires legal action masks')
            dist = distribution(self, obs, available_actions)
            action = torch.as_tensor(action, device=self.tpdv['device']).long()
            logp = dist.log_probs(action)
            entropy = dist.entropy()
            if active_masks is not None:
                active = torch.as_tensor(active_masks, **self.tpdv).squeeze(-1)
                entropy = (entropy*active).sum()/active.sum()
            else:
                entropy = entropy.mean()
            # These two upstream diagnostics are omitted from our published metrics.
            # They do not enter the update; KL is overridden below.
            return logp, entropy, dist.probs, torch.zeros_like(dist.probs)

    class CategoricalMACPO(R_MACTRPO_CPO):
        def kl_divergence(self, obs, rnn_states, action, masks, available_actions,
                          active_masks, new_actor, old_actor):
            return categorical_kl(new_actor, old_actor, obs, available_actions)

        def conjugate_gradient(self, actor, obs, rnn_states, action, masks,
                               available_actions, active_masks, b, nsteps,
                               residual_tol=1e-10):
            return conjugate_gradient(lambda p: self.fisher_vector_product(
                actor, obs, rnn_states, action, masks, available_actions, active_masks, p),
                b, nsteps, residual_tol)

    return CategoricalActor, CategoricalMACPO
