"""Uniform train/evaluate entry point for pinned paper baseline integrations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import yaml
from cage4_protocol.store import RunWriter, atomic_json
from ..envs.adapter import Cage4Adapter
from ..envs.observations import ENCODER
from ..experiments.runner import save_checkpoint
from ..recording.telemetry import frame
from ..safety.costs import COST_VERSION, measure, summarize
from ..safety.calibrate import CALIBRATION_SEEDS, VALIDATION_SEEDS
from . import ALGORITHMS


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    required = {'algorithm', 'seed', 'episodes', 'steps', 'obs_dim', 'reward_scale', 'checkpoint_every'}
    if not isinstance(c, dict) or not required <= c.keys():
        raise ValueError(f'Missing config fields: {required - (c.keys() if isinstance(c, dict) else set())}')
    allowed = required | {'hyperparameters', 'record_replay', 'snapshot_every', 'deterministic_eval', 'budget_level'}
    if set(c) - allowed: raise ValueError(f'Unknown config fields: {set(c)-allowed}')
    for flag in ('record_replay', 'deterministic_eval'):
        if flag in c and type(c[flag]) is not bool: raise ValueError(f'Invalid {flag}')
    if c['algorithm'] not in ALGORITHMS: raise ValueError('Unsupported algorithm')
    for name, low, high in [('seed', 1, 1000), ('episodes', 1, 50000), ('steps', 10, 1000),
                            ('obs_dim', 64, 8192), ('checkpoint_every', 1, 50000)]:
        if type(c[name]) is not int or not low <= c[name] <= high:
            raise ValueError(f'Invalid {name}')
    if c.get('budget_level', 'main') not in ('main', 'strict', 'loose'): raise ValueError('Invalid budget_level')
    if type(c['reward_scale']) not in (int, float) or not 0 < c['reward_scale'] <= 1: raise ValueError('Invalid reward_scale')
    if c.get('snapshot_every', 50) < 1: raise ValueError('Invalid snapshot_every')
    return c


def read_budget(path, steps, smoke=False, level='main'):
    data = json.loads(Path(path).read_text())
    if data.get('schema') != 'cage4.budget-calibration/1' or data.get('cost_version') != COST_VERSION:
        raise ValueError('Incompatible budget schema')
    if data.get('status') != 'calibrated': raise ValueError('Budget did not pass independent validation')
    if not smoke and steps != data['configured_steps']:
        raise ValueError('Horizon differs from calibration; recalibrate or use --smoke for interface tests')
    rate = data['budgets'][level+'_rate']
    if not 0 <= rate <= 1: raise ValueError('Invalid budget rate')
    return data, rate


def run(config, runs, upstreams, budget, mode='train', checkpoint=None, seeds=None, smoke=False):
    import torch
    from .backend import AuthorBaseline, fingerprint
    c = dict(config)
    budget_data, rate = read_budget(budget, c['steps'], smoke, c.get('budget_level', 'main'))
    if mode == 'evaluate' and checkpoint is None: raise ValueError('Evaluation requires a checkpoint')
    if mode == 'train' and checkpoint: raise ValueError('Resume is not implemented; checkpoint is evaluation only')
    if mode == 'evaluate':
        if not seeds or len(set(seeds)) != len(seeds): raise ValueError('Explicit unique evaluation seeds required')
        protected = set(CALIBRATION_SEEDS + VALIDATION_SEEDS)
        if any(type(s) is not int or not 0 <= s < 2**32 or s in protected for s in seeds):
            raise ValueError('Invalid evaluation seed or overlap with calibration/validation')
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.set_num_threads(1)
    adapter = Cage4Adapter(c['seed'], c['steps'], 'FiniteStateRedAgent', c['obs_dim'])
    from CybORG import __file__ as cyborg_file
    reward_file = Path(cyborg_file).parent/'Shared'/'BlueRewardMachine.py'
    if hashlib.sha256(reward_file.read_bytes()).hexdigest() != budget_data['reward_source_sha256']:
        raise ValueError('Simulator reward source differs from calibration; recalibrate first')
    inputs = adapter.reset()
    policy = AuthorBaseline(inputs, c, upstreams, rate)
    source_args = vars(policy.args).copy()
    # No private source paths are included in provenance.
    metadata = dict(mode=mode, algorithm=c['algorithm'], encoder=ENCODER, cost_version=COST_VERSION,
                    upstream=policy.provenance, safety_budget_rate=rate, safety_budget_level=c.get('budget_level', 'main'),
                    protocol='cage4.paper-common-observation/1', smoke=smoke,
                    reward_semantics='official_team_return = mean of five shared Blue returns, all executed steps',
                    critic_information='own Blue observation' if c['algorithm']=='ippo' else 'concatenated five Blue observations; no Ground Truth',
                    parameter_sharing=False, evaluation_seeds=seeds or [],
                    budget_sha256=hashlib.sha256(Path(budget).read_bytes()).hexdigest())
    payload = None
    if checkpoint:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['encoder'] != ENCODER or payload['config'] != c or payload['upstream'] != policy.provenance:
            raise ValueError('Checkpoint/config/source provenance mismatch; evaluate with training config')
        if payload['budget_sha256'] != metadata['budget_sha256']: raise ValueError('Budget changed since training')
        if set(seeds) & set(payload['training_episode_seeds']): raise ValueError('Train/test seed overlap')
        policy.load(payload['policy'])
        metadata['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    eval_weights = {a: fingerprint(p.actor) for a,p in policy.policies.items()} if mode == 'evaluate' else None
    writer = RunWriter(runs, c, metadata)
    atomic_json(writer.path/'budget.json', budget_data)
    atomic_json(writer.path/'upstream-arguments.json', source_args)
    episode_seeds = seeds if mode == 'evaluate' else [100000 + c['seed']*100000 + i for i in range(c['episodes'])]
    rows = []; trained = []
    try:
        for episode, seed in enumerate(episode_seeds, 1):
            if mode == 'evaluate':
                random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            inputs = adapter.reset(seed); policy.begin_episode()
            scores = {a: 0. for a in inputs}; costs = []; rewards_once = []
            if c.get('record_replay', True): writer.record('frames', episode, 0, frame(adapter, episode, 0))
            terminal = False
            for step in range(1, c['steps']+1):
                actions = policy.act(inputs, training=mode=='train', deterministic=c.get('deterministic_eval', True))
                labels = {a: {'index': x, 'label': inputs[a].labels[x], 'legal_mask': True} for a,x in actions.items()}
                inputs, rewards, terminated, truncated = adapter.step(actions)
                cost = measure(adapter.base); costs.append(cost)
                if mode == 'train': policy.observe(rewards, cost['cost'])
                for a in scores: scores[a] += rewards[a]
                rewards_once.append(float(np.mean(list(rewards.values()))))
                terminal = not inputs or all(terminated.get(a,False) or truncated.get(a,False) for a in scores)
                if c.get('record_replay', True):
                    # All executed actions/observations replayed; full topology sparse.
                    if step == 1 or terminal or step % c.get('snapshot_every', 50) == 0:
                        data = frame(adapter, episode, step, labels)
                    else:
                        from ..recording.telemetry import safe, red_telemetry
                        data = dict(blue=labels, blue_observations=safe(adapter.raw), red=red_telemetry(adapter.base))
                    data.update(rewards=rewards, safety=cost, seed=seed)
                    writer.record('frames', episode, step, data)
                if step % 10 == 0: writer.status(episode=episode, step=step, seed=seed)
                if terminal: break
            if not terminal: raise RuntimeError('Rollout cut before true terminal; refusing zero-bootstrap update')
            losses = policy.update(c['reward_scale']) if mode == 'train' else {}
            summary = summarize(costs)
            row = dict(episode_seed=seed, seed=c['seed'], steps=step, agents=scores, losses=losses,
                       team_return=sum(scores.values()), official_team_return=sum(rewards_once),
                       official_script_compat_return=sum(rewards_once[:-1]), safety=summary,
                       safety_budget_rate=rate, budget_excess=summary['mean_cost']-rate,
                       budget_violated=summary['mean_cost'] > rate)
            writer.record('metrics', episode, step, row); rows.append(row)
            if mode == 'train':
                trained.append(seed)
                if episode % c['checkpoint_every'] == 0 or episode == len(episode_seeds):
                    save_checkpoint(writer.path/'checkpoints'/f'episode_{episode:06d}.pt',
                                    dict(encoder=ENCODER, config=c, upstream=policy.provenance,
                                         budget_sha256=metadata['budget_sha256'], training_episode_seeds=trained.copy(),
                                         episode=episode, policy=policy.checkpoint()))
            print(json.dumps(dict(episode=episode, seed=seed, official_team_return=row['official_team_return'],
                                  mean_cost=summary['mean_cost'], budget_rate=rate)), flush=True)
        if mode == 'evaluate':
            if eval_weights != {a: fingerprint(p.actor) for a,p in policy.policies.items()}:
                raise RuntimeError('Evaluation changed actor parameters')
            scores = [r['official_team_return'] for r in rows]
            atomic_json(writer.path/'evaluation.json', dict(
                metric='official_team_return', evaluation_seeds=episode_seeds,
                by_seed={str(r['episode_seed']): r['official_team_return'] for r in rows},
                mean=float(np.mean(scores)), sample_std=float(np.std(scores, ddof=1)) if len(scores)>1 else None,
                n_seeds=len(scores), mean_cost=float(np.mean([r['safety']['mean_cost'] for r in rows])),
                violation_fraction=float(np.mean([r['budget_violated'] for r in rows])), complete=True))
        writer.finish('finished')
    except KeyboardInterrupt:
        writer.finish('stopped'); raise
    except BaseException as exc:
        import traceback
        writer.finish('error', error=str(exc), traceback=traceback.format_exc()); raise
    return writer.path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['train', 'evaluate'])
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--runs', type=Path, required=True)
    p.add_argument('--upstreams', type=Path, required=True)
    p.add_argument('--budget', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--seeds', type=int, nargs='+')
    p.add_argument('--smoke', action='store_true', help='Interface test only; allow a shorter uncalibrated horizon')
    a = p.parse_args()
    print(run(load_config(a.config), a.runs, a.upstreams, a.budget, a.mode, a.checkpoint, a.seeds, a.smoke))


if __name__ == '__main__': main()
