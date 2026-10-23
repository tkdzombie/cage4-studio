"""Paired raw-action PPO checkpoints, common scenario/exploration random numbers."""
import argparse,json,pickle,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from primitive_env import PrimitiveEnv,UPSTREAM
from ray.rllib.policy.policy import Policy
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--seeds',default='54001,54002');p.add_argument('--endpoints',default='initial,latest');p.add_argument('--output',default='flat-comparison.json');p.add_argument('--protocol');a=p.parse_args()
config_path=Path(a.run)/'config.json'
run_config=json.loads(config_path.read_text()) if config_path.exists() else {}
recorded_protocol=run_config.get('protocol')
if a.protocol and recorded_protocol and a.protocol!=recorded_protocol:raise ValueError('Evaluation protocol differs from saved run; use an explicitly named transfer experiment')
a.protocol=a.protocol or recorded_protocol or 'flat-history-correct'
torch.set_num_threads(1)
e=PrimitiveEnv({'seed':321,'protocol':a.protocol})
policies={f'blue_agent_{i}':Policy.from_checkpoint(str(UPSTREAM/'models/train_marl/iter_199/policies'/f'Agent{i}')) for i in range(5)}
author={k:{n:np.array(v,copy=True) for n,v in policy.get_weights().items()} for k,policy in policies.items()}
rows=[];checkpoint_metadata={}
reference='author-restricted' if a.protocol=='flat-full-actions-history-correct' else 'author'
for endpoint in [reference]+a.endpoints.split(','):
    for i in range(5):
        k=f'blue_agent_{i}'
        if endpoint.startswith('author'):weights=author[k]
        else:
            policy_dir=Path(a.run)/endpoint/'policies'/f'Agent{i}'
            raw=(policy_dir/'policy_state.pkl').read_bytes();state=pickle.loads(raw);weights=state['weights']
            # Exercise the real saved Policy API, not just parameter assignment.
            policies[k]=Policy.from_checkpoint(str(policy_dir))
            checkpoint_metadata.setdefault(endpoint,{})[k]={'sha256':hashlib.sha256(raw).hexdigest(),'global_timestep':state.get('global_timestep')}
        assert {n:np.shape(v) for n,v in weights.items()}=={n:np.shape(v) for n,v in policies[k].get_weights().items()}
        policies[k].set_weights(weights)
    for seed in map(int,a.seeds.split(',')):
        torch.manual_seed(seed+710000);np.random.seed(seed+710000);random.seed(seed+710000)
        obs,_=e.reset(seed=seed);counts={};free={};components={};forced=0;start=time.time()
        while True:
            acts={}
            for k,v in obs.items():
                actor_obs=v
                if endpoint=='author-restricted':
                    mask=v['action_mask'].copy()
                    for idx,label in enumerate(e.inner.action_labels(k)):
                        if label.startswith(('BlockTraffic','AllowTraffic')):mask[idx]=0
                    actor_obs=dict(v,action_mask=mask)
                acts[k]=int(policies[k].compute_single_action(actor_obs,explore=True)[0])
            for k,act in acts.items():
                command=e.inner.action_labels(k)[act].split()[0]
                assert command not in ['[Invalid]','[Padding]']
                counts[command]=counts.get(command,0)+1
                if sum(obs[k]['action_mask'])>1:free[command]=free.get(command,0)+1
                else:forced+=1
            obs,rew,term,trunc,info=e.step(acts)
            def scalar(value):return sum(scalar(x) for x in value.values()) if isinstance(value,dict) else float(value)
            for component,value in e.inner.env.get_rewards()['Blue'].items():components[component]=components.get(component,0.0)+scalar(value)
            if term['__all__'] or trunc['__all__']:break
        row={'endpoint':endpoint,'seed':seed,'return':e.team_return,'ticks':e.ticks,'seconds':time.time()-start,'primitive_actions':counts,'free_actions':free,'forced_sleep':forced,'reward_components':components}
        assert np.isclose(sum(components.values()),e.team_return)
        rows.append(row);print(json.dumps(row),flush=True)
json.dump({'protocol':a.protocol,'run':Path(a.run).name,'exploration_seed':'scenario+710000 reset per episode','reference_note':'author-restricted preserves the authors original no-traffic policy action subset in the same repaired simulator','rows':rows,'checkpoint_metadata':checkpoint_metadata,'means':{k:float(np.mean([r['return'] for r in rows if r['endpoint']==k])) for k in sorted({r['endpoint'] for r in rows})}},open(a.output,'w'),indent=2)
