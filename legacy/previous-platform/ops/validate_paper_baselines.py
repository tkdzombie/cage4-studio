"""Run real short training/evaluation for every author-source integration.

Invoke using the research interpreter, not on a machine without CybORG/Torch.
No performance claim is made by this interface/weight-update test.
"""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from cage4_protocol.store import atomic_json
from cage4_baselines.paper.__main__ import run, load_config, read_budget
from cage4_baselines.paper.backend import ALGORITHMS


def validate(runs, upstreams, budget, output, full=False, algorithms=ALGORITHMS):
    report = {'schema': 'cage4.paper-validation/1', 'scope': 'full-horizon integration; not performance' if full else 'real simulator smoke; not paper performance', 'algorithms': {}}
    for name in algorithms:
        if name not in ALGORITHMS: raise ValueError('Unknown algorithm')
        config = dict(algorithm=name, seed=101, episodes=2, steps=20,
                      reward_scale=.01, rollout_episodes=2, checkpoint_every=1, record_replay=True,
                      snapshot_every=10, deterministic_eval=True,
                      hyperparameters={'ppo_epoch': 2, 'hidden_size': 32})
        if full:
            config=load_config(Path(__file__).resolve().parents[1]/'cage4-baselines/configs/paper'/f'{name}.yaml')
            config.update(episodes=8,checkpoint_every=8,record_replay=False)
        from cage4_baselines.envs.adapter import Cage4Adapter
        from cage4_baselines.paper.backend import AuthorBaseline
        from cage4_baselines.paper.validation import check_bridge
        env=Cage4Adapter(seed=42,steps=20)
        inputs=env.reset()
        _,rate=read_budget(budget,config['steps'],smoke=not full)
        bridge=check_bridge(AuthorBaseline(inputs,config,upstreams,rate),inputs)
        train = run(config, runs, upstreams, budget, smoke=not full)
        records = [json.loads(p.read_text()) for p in sorted((train/'metrics').rglob('*.json'))]
        cp = sorted((train/'checkpoints').glob('*.pt'))[-1]
        # Safe loading must work on the actual supported Torch version.
        payload = torch.load(cp, map_location='cpu', weights_only=True)
        checkpoint_hash = hashlib.sha256(cp.read_bytes()).hexdigest()
        evaluation = run(config, runs, upstreams, budget, mode='evaluate', checkpoint=cp,
                         seeds=[53001, 53002], smoke=not full)
        if hashlib.sha256(cp.read_bytes()).hexdigest() != checkpoint_hash:
            raise AssertionError('Evaluation modified saved checkpoint')
        updates = {}
        for a in payload['policy']:
            keys = ['actor_changed', 'critic_changed']
            if name in ('macpo', 'mappo_lagrangian'): keys.append('cost_critic_changed')
            updates[a] = {k: any(r['losses'].get(a,{}).get(k,False) for r in records) for k in keys}
            # A CPO line search may legitimately reject an actor step. Report it.
            for k,v in updates[a].items():
                if not v and not (name=='macpo' and k=='actor_changed'):
                    raise AssertionError(f'{name}/{a}/{k}: weights never changed')
        report['algorithms'][name] = dict(bridge=bridge, train_run=train.name, evaluation_run=evaluation.name,
                                          weight_updates=updates, train_episodes=len(records),
                                          evaluation=json.loads((evaluation/'evaluation.json').read_text()),
                                          transitions=sum(r['steps'] for r in records),
                                          optimizer_episodes=[r['episode'] for r in records if r['optimizer_update']])
        atomic_json(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'upstreams', 'budget', 'output'): p.add_argument('--'+key, type=Path, required=True)
    p.add_argument('--full',action='store_true',help='Eight complete 500-config-step episodes with research profile')
    p.add_argument('--algorithms',nargs='+',choices=ALGORITHMS,default=ALGORITHMS)
    a = p.parse_args()
    validate(a.runs, a.upstreams, a.budget, a.output, a.full, a.algorithms)
