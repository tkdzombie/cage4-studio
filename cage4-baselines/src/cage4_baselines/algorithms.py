"""Extensible registry. IPPO and MAPPO are deliberately unimplemented."""
import importlib
import numpy as np
REGISTRY = {}
def register(name):
    def decorate(cls):
        if name in REGISTRY:
            raise ValueError('Duplicate algorithm: '+name)
        REGISTRY[name] = cls
        return cls
    return decorate

def create(name, inputs, config):
    for module in config.get('plugins',[]):
        importlib.import_module(module)  # explicit trusted research configuration only
    if name not in REGISTRY:
        raise ValueError(f'Unknown/unimplemented algorithm {name}; available: {sorted(REGISTRY)}')
    return REGISTRY[name](inputs,config)

@register('random')
class RandomPolicy:
    def __init__(self, inputs, config):
        self.rng = np.random.default_rng(config['seed'])
    def act(self, inputs, training=False):
        return {a:int(self.rng.choice(np.flatnonzero(x.mask))) for a,x in inputs.items()}
    def observe(self, rewards):
        pass
    def update(self):
        return {}
    def checkpoint(self):
        return {}
    def load(self, data):
        pass

@register('sleep')
class SleepPolicy(RandomPolicy):
    def act(self, inputs, training=False):
        result = {}
        for a,x in inputs.items():
            candidates = [i for i,v in enumerate(x.labels) if x.mask[i] and v.startswith('Sleep')]
            if not candidates:
                raise RuntimeError('Official Sleep action missing')
            result[a] = candidates[0]
        return result

@register('actor_critic')
class ActorCriticPolicy:
    """Five independent episodic REINFORCE policies with learned value baselines.

    Shared actor/critic trunk within each agent; no parameter sharing across agents.
    Monte Carlo returns are cut at the configured horizon. This is not PPO.
    """
    def __init__(self, inputs, config):
        import torch
        from torch import nn
        self.torch,self.config = torch,config
        self.device = torch.device(config['device'])
        torch.manual_seed(config['seed'])
        torch.set_num_threads(min(8,config.get('torch_threads',4)))
        self.models,self.optimizers = {},{}
        self.spaces = {a:(len(x.observation),len(x.mask)) for a,x in inputs.items()}
        for a,(dim,n) in self.spaces.items():
            # Last output is the value estimate; other outputs are action logits.
            model = nn.Sequential(nn.Linear(dim,128),nn.Tanh(),nn.Linear(128,128),
                                  nn.Tanh(),nn.Linear(128,n+1)).to(self.device)
            self.models[a] = model
            self.optimizers[a] = torch.optim.Adam(model.parameters(),lr=config['lr'])
        self.memory = {a:[] for a in inputs}
    def act(self, inputs, training=False):
        torch = self.torch
        actions = {}
        with torch.set_grad_enabled(training):
            for a,x in inputs.items():
                values = self.models[a](torch.as_tensor(x.observation,device=self.device))
                mask = torch.as_tensor(x.mask,device=self.device)
                if len(mask) != len(values)-1 or not bool(mask.any()):
                    raise ValueError('Invalid policy action mask')
                dist = torch.distributions.Categorical(logits=values[:-1].masked_fill(~mask,-1e9))
                action = dist.sample()
                actions[a] = int(action.item())
                if training:
                    self.memory[a].append([dist.log_prob(action),values[-1],dist.entropy(),None])
        return actions
    def observe(self, rewards):
        for a,mem in self.memory.items():
            if mem:
                mem[-1][-1] = rewards[a]
    def update(self):
        torch = self.torch
        metrics = {}
        for a,mem in self.memory.items():
            if not mem:
                continue
            g,returns = 0.,[]
            for *_,reward in reversed(mem):
                g = float(reward)+self.config['gamma']*g
                returns.append(g)
            ret = torch.tensor(list(reversed(returns)),device=self.device)
            logp = torch.stack([v[0] for v in mem])
            values = torch.stack([v[1] for v in mem])
            entropy = torch.stack([v[2] for v in mem]).mean()
            actor = -(logp*(ret-values.detach())).mean()
            critic = 0.5*(ret-values).square().mean()
            loss = actor+critic-self.config['entropy_coef']*entropy
            self.optimizers[a].zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.models[a].parameters(),1.)
            self.optimizers[a].step()
            metrics[a] = {k:float(v.detach().cpu()) for k,v in
                          [('actor_loss',actor),('critic_loss',critic),('entropy',entropy),('loss',loss)]}
            mem.clear()
        return metrics
    def checkpoint(self):
        return {'spaces':self.spaces,'models':{a:m.state_dict() for a,m in self.models.items()},
                'optimizers':{a:o.state_dict() for a,o in self.optimizers.items()}}
    def load(self, data):
        if data['spaces'] != self.spaces:
            raise ValueError('Checkpoint spaces differ from current environment')
        for a,m in self.models.items():
            m.load_state_dict(data['models'][a])
