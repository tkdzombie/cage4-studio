"""Restore actual policies and evaluate common full scenarios in raw reward units."""
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from ray.rllib.policy.policy import Policy
from cage4.environment import PrimitiveEnv


def evaluate(run, seeds, endpoints, output=None):
    run = Path(run).resolve()
    config = json.loads((run/'config.json').read_text())
    protocol = config['protocol']
    torch.set_num_threads(1)
    env = PrimitiveEnv({'seed': 321, 'protocol': protocol})
    rows = []
    metadata = {}
    for endpoint in endpoints:
        policies = {}
        metadata[endpoint] = {}
        for i in range(5):
            folder = run/endpoint/'policies'/f'Agent{i}'
            policies[f'blue_agent_{i}'] = Policy.from_checkpoint(str(folder))
            digest = hashlib.sha256((folder/'policy_state.pkl').read_bytes()).hexdigest()
            metadata[endpoint][f'Agent{i}'] = digest
        for seed in seeds:
            torch.manual_seed(seed+710000)
            np.random.seed(seed+710000)
            random.seed(seed+710000)
            obs,_ = env.reset(seed=seed)
            counts,components = {},{}
            started = time.time()
            while True:
                actions = {agent: int(policies[agent].compute_single_action(value, explore=True)[0])
                           for agent,value in obs.items()}
                for agent,action in actions.items():
                    command = env.inner.action_labels(agent)[action].split()[0]
                    counts[command] = counts.get(command,0)+1
                obs,_,term,trunc,_ = env.step(actions)
                def scalar(value):
                    return sum(scalar(v) for v in value.values()) if isinstance(value,dict) else float(value)
                for component,value in env.inner.env.get_rewards()['Blue'].items():
                    components[component] = components.get(component,0.0)+scalar(value)
                if term['__all__'] or trunc['__all__']:
                    break
            assert env.ticks == 499 and np.isclose(sum(components.values()),env.team_return)
            row = {'endpoint': endpoint, 'seed': seed, 'return': env.team_return,
                   'ticks': env.ticks, 'seconds': time.time()-started,
                   'primitive_actions': counts, 'reward_components': components}
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {'protocol': protocol, 'run': run.name, 'exploration_seed': 'scenario+710000 reset per episode',
              'rows': rows, 'checkpoint_sha256': metadata,
              'means': {endpoint: float(np.mean([r['return'] for r in rows if r['endpoint']==endpoint]))
                        for endpoint in endpoints}}
    target = Path(output) if output else run/'evaluation.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(result,indent=2))
    temporary.replace(target)
    return result
