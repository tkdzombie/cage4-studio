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
from cage4_baselines.paper.__main__ import run
from cage4_baselines.paper.backend import ALGORITHMS


def validate(runs, upstreams, budget, output):
    report = {'schema': 'cage4.paper-validation/1', 'scope': 'real simulator smoke; not paper performance', 'algorithms': {}}
    for name in ALGORITHMS:
        config = dict(algorithm=name, seed=101, episodes=2, steps=20, obs_dim=128,
                      reward_scale=.01, checkpoint_every=1, record_replay=True,
                      snapshot_every=10, deterministic_eval=True,
                      hyperparameters={'ppo_epoch': 2, 'hidden_size': 32})
        train = run(config, runs, upstreams, budget, smoke=True)
        records = [json.loads(p.read_text()) for p in sorted((train/'metrics').rglob('*.json'))]
        cp = sorted((train/'checkpoints').glob('*.pt'))[-1]
        # Safe loading must work on the actual supported Torch version.
        payload = torch.load(cp, map_location='cpu', weights_only=True)
        checkpoint_hash = hashlib.sha256(cp.read_bytes()).hexdigest()
        evaluation = run(config, runs, upstreams, budget, mode='evaluate', checkpoint=cp,
                         seeds=[53001, 53002], smoke=True)
        if hashlib.sha256(cp.read_bytes()).hexdigest() != checkpoint_hash:
            raise AssertionError('Evaluation modified saved checkpoint')
        updates = {}
        for a in payload['policy']:
            keys = ['actor_changed', 'critic_changed']
            if name in ('macpo', 'mappo_lagrangian'): keys.append('cost_critic_changed')
            updates[a] = {k: any(r['losses'][a][k] for r in records) for k in keys}
            # A CPO line search may legitimately reject an actor step. Report it.
            for k,v in updates[a].items():
                if not v and not (name=='macpo' and k=='actor_changed'):
                    raise AssertionError(f'{name}/{a}/{k}: weights never changed')
        report['algorithms'][name] = dict(train_run=train.name, evaluation_run=evaluation.name,
                                          weight_updates=updates, train_episodes=len(records),
                                          evaluation=json.loads((evaluation/'evaluation.json').read_text()))
        atomic_json(output, report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('runs', 'upstreams', 'budget', 'output'): p.add_argument('--'+key, type=Path, required=True)
    a = p.parse_args()
    validate(a.runs, a.upstreams, a.budget, a.output)
