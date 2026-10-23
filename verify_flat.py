import json, numpy as np
from primitive_env import PrimitiveEnv
np.random.seed(99)
e=PrimitiveEnv({'seed':321,'protocol':'flat-full-actions-history-correct'})
def forbidden(*args,**kwargs):raise AssertionError('Hidden infection truth helper was used')
e.inner.infection_status=forbidden;e.inner._get_files_from_state=forbidden
obs,_=e.reset(seed=54992);counts={};forced=0
while True:
    acts={a:int(np.random.choice(np.flatnonzero(o['action_mask']))) for a,o in obs.items()}
    for a,k in acts.items():
        cmd=e.inner.action_labels(a)[k].split()[0];assert cmd not in ['[Invalid]','[Padding]']
        counts[cmd]=counts.get(cmd,0)+1;forced+=int(sum(obs[a]['action_mask'])==1)
    obs,rew,term,trunc,info=e.step(acts)
    raw=e.inner.env.get_rewards()['Blue']
    def total(v):return sum(total(x) for x in v.values()) if isinstance(v,dict) else float(v)
    assert all(r==total(raw) for r in rew.values())
    assert all(e.observation_space[a].contains(o) and np.any(o['action_mask']) for a,o in obs.items())
    if term['__all__'] or trunc['__all__']:
        assert all(term.values()) and not any(trunc.values());break
assert e.ticks==499
print(json.dumps({'ticks':e.ticks,'team_return':e.team_return,'primitive_actions':counts,'singleton_masks':forced,'observation_spaces':{a:e.observation_space[a]['observations'].shape for a in obs},'actions':{a:e.action_space[a].n for a in obs}},default=int),flush=True)
