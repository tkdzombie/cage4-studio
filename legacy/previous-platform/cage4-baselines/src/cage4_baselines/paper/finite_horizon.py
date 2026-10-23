"""Explicit finite-horizon OT variants of the pinned constrained baselines.

MC returns and episode-scaled raw advantages keep the event surrogate in
probability units. Original actor/critic networks and CPO solver are reused.
No original paper theorem or exact on-policy safety guarantee is claimed.
"""
from copy import copy
from types import MethodType
import numpy as np
import torch
from .backend import AuthorBaseline, array, fingerprint
from .discrete_macpo import distribution
from ..safety.continuity_events import undiscounted_returns


def preserve_dual(original):
    def update(trainer,*args,**kwargs):
        before=trainer.lamda_lagr
        try:return original(*args,**kwargs)
        finally:trainer.lamda_lagr=before
    return update


def finite_cpo_rescale(original):
    def update(trainer,*args,**kwargs):
        before=trainer.gamma
        # In this pinned trpo_update, gamma is used ONLY in (1-gamma)*violation.
        # No buffer/return calculation occurs here. gamma=0 locally makes that
        # multiplier one; reward and cost returns are computed explicitly below.
        trainer.gamma=0.
        try:return original(*args,**kwargs)
        finally:trainer.gamma=before
    return update


class FiniteHorizonBaseline(AuthorBaseline):
    def __init__(self,inputs,config,upstreams,risk_limit):
        if config['algorithm'] not in ('macpo','mappo_lagrangian'):
            raise ValueError('OT tests only support the constrained baselines')
        super().__init__(inputs,config,upstreams,risk_limit)
        for trainer in self.trainers.values():
            trainer.safety_bound=risk_limit
            if self.name=='macpo':
                trainer._max_lin_constraint_val=risk_limit
                trainer.trpo_update=MethodType(finite_cpo_rescale(trainer.trpo_update),trainer)
            else:
                trainer.ppo_update=MethodType(preserve_dual(trainer.ppo_update),trainer)

    def update(self,reward_scale):
        T=len(self.history);terminals=[x[3] for x in self.history];B=sum(terminals)
        if not B or not terminals[-1]:raise ValueError('Only complete episodes may update')
        costs=[x[2] for x in self.history]
        if any(c not in (0.,1.) for c in costs):raise ValueError('Expected binary first-hit costs')
        event_returns=np.asarray(undiscounted_returns(costs,terminals),np.float32)
        if any(x not in (0.,1.) for x in event_returns):raise ValueError('At most one event per episode')
        risk=sum(costs)/B;scale=T/B;buffers={}
        for a in self.names:
            args=copy(self.args);args.episode_length=T
            b=self.Buffer(args,*self.spaces[a])
            for t,(records,rewards,cost,terminal) in enumerate(self.history):
                r=records[a]
                for target,key in [('obs','obs'),('share_obs','cent'),('available_actions','available'),('actions','action'),('action_log_probs','logp'),('value_preds','value'),('cost_preds','cost_value')]:
                    getattr(b,target)[t]=r[key]
                b.masks[t+1]=0. if terminal else 1.
                b.costs[t]=cost;b.rewards[t]=rewards[a]*reward_scale
            rewards=[x[1][a]*reward_scale for x in self.history]
            b.returns[:-1,0,0]=undiscounted_returns(rewards,terminals)
            b.cost_returns[:-1,0,0]=event_returns
            b.aver_episode_costs.fill(risk)
            buffers[a]=b
        factor=np.ones((T,1,1),np.float32);metrics={}
        for a in [self.names[i] for i in torch.randperm(len(self.names)).tolist()]:
            p,tr,b=self.policies[a],self.trainers[a],buffers[a]
            before={k:fingerprint(getattr(p,k)) for k in ('actor','critic','cost_critic')}
            old_logp=self.logp(a,b);b.update_factor(factor)
            with torch.no_grad():
                old=distribution(p.actor,b.obs[:-1,0],b.available_actions[:-1,0])
                old=torch.distributions.Categorical(logits=old.logits.detach().clone())
                mse_before=float(((p.get_cost_values(b.share_obs[:-1,0],b.rnn_states_cost[:-1,0],b.masks[:-1,0])-torch.from_numpy(b.cost_returns[:-1,0]))**2).mean())
            # mean over transitions times T/B equals a sum per complete episode.
            reward_adv=(b.returns[:-1]-b.value_preds[:-1])*scale
            cost_adv=(b.cost_returns[:-1]-b.cost_preds[:-1])*scale
            risk_surrogate_before=risk+float(((factor-1.)*cost_adv).mean())
            if self.name=='macpo':b.aver_episode_costs.fill(risk_surrogate_before)
            entropy=[];tr.prep_training()
            lambda_before=float(tr.lamda_lagr) if self.name=='mappo_lagrangian' else None
            for _ in range(tr.ppo_epoch):
                for sample in b.feed_forward_generator(reward_adv,tr.num_mini_batch,cost_adv=cost_adv):
                    out=tr.trpo_update(sample) if self.name=='macpo' else tr.ppo_update(sample)
                    entropy.append(float(out[5 if self.name=='macpo' else 3].detach()))
            if self.name=='mappo_lagrangian':
                tr.lamda_lagr=torch.tensor(max(0.,lambda_before+tr.lagrangian_coef*(risk-self.rate)))
            factor*=array(torch.exp(self.logp(a,b)-old_logp)).reshape(T,1,1)
            with torch.no_grad():
                new=distribution(p.actor,b.obs[:-1,0],b.available_actions[:-1,0])
                kl=float(torch.distributions.kl_divergence(old,torch.distributions.Categorical(logits=new.logits)).mean())
                mse=float(((p.get_cost_values(b.share_obs[:-1,0],b.rnn_states_cost[:-1,0],b.masks[:-1,0])-torch.from_numpy(b.cost_returns[:-1,0]))**2).mean())
            values=dict(episode_risk=risk,event_budget=self.rate,episodes=B,transitions=T,
                        surrogate_episode_scale=scale,cost_mse_before=mse_before,cost_mse_after=mse,
                        actual_policy_kl=kl,entropy=float(np.mean(entropy)),risk_surrogate_before=risk_surrogate_before)
            if lambda_before is not None:values.update(lambda_before=lambda_before,lambda_after=float(tr.lamda_lagr))
            if self.name=='macpo':values.update(tr.last_step_diagnostics)
            for key,previous in before.items():
                module=getattr(p,key)
                if any(not torch.isfinite(v).all() for v in module.parameters()):raise FloatingPointError('Nonfinite '+a+'/'+key)
                values[key+'_changed']=fingerprint(module)!=previous
            if not all(np.isfinite(v) for v in values.values()):raise FloatingPointError('Nonfinite OT diagnostic')
            metrics[a]=values
        self.history=[]
        return metrics
