"""Uniform train/evaluate entry point for pinned paper baseline integrations."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import shutil
import time
from contextlib import nullcontext
import numpy as np
import yaml
from cage4_protocol.store import RunWriter, atomic_json
from ..envs.adapter import Cage4Adapter
from ..envs.observations import ENCODER
from ..experiments.runner import save_checkpoint
from ..recording.telemetry import frame
from ..safety.costs import COST_VERSION, measure, summarize
from ..safety.calibrate import CALIBRATION_SEEDS, VALIDATION_SEEDS
from ..safety.budgets import read_budget
from . import ALGORITHMS


def load_config(path):
    c = yaml.safe_load(Path(path).read_text())
    required = {'algorithm', 'seed', 'episodes', 'steps', 'reward_scale', 'checkpoint_every'}
    if not isinstance(c, dict) or not required <= c.keys():
        raise ValueError(f'Missing config fields: {required - (c.keys() if isinstance(c, dict) else set())}')
    allowed = required | {'hyperparameters', 'record_replay', 'snapshot_every', 'deterministic_eval', 'budget_level', 'rollout_episodes', 'observation', 'num_envs', 'torch_threads'}
    if set(c) - allowed: raise ValueError(f'Unknown config fields: {set(c)-allowed}')
    for flag in ('record_replay', 'deterministic_eval'):
        if flag in c and type(c[flag]) is not bool: raise ValueError(f'Invalid {flag}')
    if c['algorithm'] not in ALGORITHMS: raise ValueError('Unsupported algorithm')
    for name, low, high in [('seed', 1, 1000), ('episodes', 1, 50000), ('steps', 10, 1000),
                            ('checkpoint_every', 1, 50000)]:
        if type(c[name]) is not int or not low <= c[name] <= high:
            raise ValueError(f'Invalid {name}')
    if c.get('budget_level', 'main') not in ('main', 'strict', 'loose'): raise ValueError('Invalid budget_level')
    if type(c['reward_scale']) not in (int, float) or not 0 < c['reward_scale'] <= 1: raise ValueError('Invalid reward_scale')
    if c.get('snapshot_every', 50) < 1: raise ValueError('Invalid snapshot_every')
    if type(c.get('rollout_episodes', 8)) is not int or not 1 <= c.get('rollout_episodes',8) <= 128:
        raise ValueError('Invalid rollout_episodes')
    for key,high in (('num_envs',28),('torch_threads',28)):
        if type(c.get(key,1)) is not int or not 1<=c.get(key,1)<=high:raise ValueError('Invalid '+key)
    from ..envs.belief import observation_config
    observation_config(c.get('observation'))
    return c


def run(config, runs, upstreams, budget, mode='train', checkpoint=None, seeds=None, smoke=False, policy_factory=None):
    run_started=time.perf_counter()
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
    random.seed(c['seed']); np.random.seed(c['seed']); torch.manual_seed(c['seed']); torch.set_num_threads(c.get('torch_threads',1))
    adapter = Cage4Adapter(c['seed'], c['steps'], 'FiniteStateRedAgent', observation=c.get('observation'), include_graph=policy_factory is not None)
    from CybORG import __file__ as cyborg_file
    reward_file = Path(cyborg_file).parent/'Shared'/'BlueRewardMachine.py'
    if hashlib.sha256(reward_file.read_bytes()).hexdigest() != budget_data['reward_source_sha256']:
        raise ValueError('Simulator reward source differs from calibration; recalibrate first')
    from .provenance import runtime_identity
    runtime = runtime_identity()
    inputs = adapter.reset()
    factory = AuthorBaseline if policy_factory is None else policy_factory
    policy = factory(inputs, c, upstreams, rate)
    source_args = vars(policy.args).copy()
    # No private source paths are included in provenance.
    metadata = dict(runtime=runtime, mode=mode, algorithm=c['algorithm'], encoder=ENCODER, cost_version=COST_VERSION,
                    upstream=policy.provenance, safety_budget_rate=rate, safety_budget_level=c.get('budget_level', 'main'),
                    protocol='cage4.paper-common-observation/3', smoke=smoke,
                    reward_semantics='official_team_return = mean of five shared Blue returns, all executed steps',
                    critic_information='own Blue observation' if c['algorithm']=='ippo' else 'concatenated five Blue observations; no Ground Truth',
                    execution=dict(num_envs=c.get('num_envs',1),torch_threads=c.get('torch_threads',1),
                                   sampler='synchronous complete episodes; frozen weights; episode-seeded RNG'),
                    parameter_sharing=False, evaluation_seeds=seeds or [],
                    budget_sha256=hashlib.sha256(Path(budget).read_bytes()).hexdigest())
    payload = None
    if checkpoint:
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        if payload['encoder'] != ENCODER or payload['config'] != c or payload['upstream'] != policy.provenance:
            raise ValueError('Checkpoint/config/source provenance mismatch; evaluate with training config')
        if payload.get('runtime') != runtime: raise ValueError('Runtime/source changed since training')
        if payload['budget_sha256'] != metadata['budget_sha256']: raise ValueError('Budget changed since training')
        if set(seeds) & set(payload['training_episode_seeds']): raise ValueError('Train/test seed overlap')
        policy.load(payload['policy'])
        metadata['checkpoint_sha256'] = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
    eval_weights = {a: fingerprint(p.actor) for a,p in policy.policies.items()} if mode == 'evaluate' else None
    if c.get('num_envs',1)>1 and policy_factory is not None:
        raise ValueError('Custom policy factory currently requires num_envs: 1')
    writer = RunWriter(runs, c, metadata)
    # Preserve the exact policy implementation so future code upgrades need not
    # invalidate evaluation of this run. Dependencies still require exact parity.
    shutil.copytree(Path(__file__).resolve().parents[1], writer.path/'source'/'cage4_baselines',
                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    atomic_json(writer.path/'observation-schema.json', adapter.observation_schema())
    atomic_json(writer.path/'spaces.json', {a:dict(observation_dim=len(x.observation), actions=len(x.mask)) for a,x in inputs.items()})
    atomic_json(writer.path/'budget.json', budget_data)
    atomic_json(writer.path/'upstream-arguments.json', source_args)
    episode_seeds = seeds if mode == 'evaluate' else [100000 + c['seed']*100000 + i for i in range(c['episodes'])]
    rows = []; trained = []; last_optimizer_episode=0; total_transitions=0
    from .rollouts import EpisodeCollector, RolloutPool
    expected_steps=budget_data['calibration'][0]['steps'] if c['steps']==budget_data['configured_steps'] else c['steps']-1
    batch_size=c.get('rollout_episodes',8) if mode=='train' else min(c.get('num_envs',1),len(episode_seeds))
    workers=min(c.get('num_envs',1),batch_size,len(episode_seeds))
    if workers>1 and policy_factory is not None:
        raise ValueError('Custom policy factory currently requires num_envs: 1')
    collector=EpisodeCollector(c,rate,expected_steps,policy,adapter)
    started=run_started
    def progress(ep,step,pid=None):
        writer.status(episode=ep,step=step,phase='collecting',worker_pid=pid,
                      completed_episodes=len(rows),total_transitions=total_transitions,
                      elapsed_seconds=time.perf_counter()-started)
    try:
      with (RolloutPool(workers,c,upstreams,rate,expected_steps) if workers>1 else nullcontext(None)) as pool:
        if pool is not None:writer.status(worker_pids=pool.worker_pids,num_envs=workers)
        start=0;pending_history=[];pending_episodes=0
        while start<len(episode_seeds):
            # Checkpoint boundaries split collection only; gradient batches stay
            # fixed and pending transitions never enter a checkpoint/resume claim.
            size=min(batch_size-pending_episodes,len(episode_seeds)-start)
            if mode=='train':size=min(size,c['checkpoint_every']-start%c['checkpoint_every'])
            jobs=[(i+1,episode_seeds[i]) for i in range(start,start+size)]
            collection_start=time.perf_counter()
            if pool is None:
                results=[];batch_history=[]
                for episode,seed in jobs:
                    result=collector.collect(episode,seed,writer.path,mode=='train',lambda ep,st:progress(ep,st))
                    results.append(result)
                    batch_history.extend(result['history'])
                pending_history.extend(batch_history)
            else:
                results=pool.collect(policy.rollout_state(),jobs,writer.path,mode=='train',progress)
                pending_history.extend(transition for result in results for transition in result['history'])
            collection_seconds=time.perf_counter()-collection_start
            if mode=='train':pending_episodes+=len(results)
            update_due=mode=='train' and (pending_episodes==batch_size or start+size==len(episode_seeds))
            policy.history=pending_history
            writer.status(phase='updating' if update_due else 'recording',episode=jobs[-1][0],step=results[-1]['row']['steps'])
            gradient_batch_episodes=pending_episodes if update_due else 0
            gradient_batch_transitions=len(pending_history) if update_due else 0
            update_start=time.perf_counter()
            losses=policy.update(c['reward_scale']) if update_due else {}
            update_seconds=time.perf_counter()-update_start
            if update_due:last_optimizer_episode=jobs[-1][0]
            if update_due:pending_history=[];pending_episodes=0
            batch_transitions=sum(result['row']['steps'] for result in results)
            batch_perf=dict(collection_seconds=collection_seconds,update_seconds=update_seconds,
                            environment_transitions=batch_transitions,collection_chunk_episodes=len(results),
                            gradient_batch_episodes=gradient_batch_episodes,gradient_batch_transitions=gradient_batch_transitions,
                            transitions_per_second=batch_transitions/max(collection_seconds+update_seconds,1e-9),
                            workers=workers,torch_threads=c.get('torch_threads',1))
            for index,result in enumerate(results):
                episode=result['episode'];row=result['row'];step=row['steps']
                total_transitions+=step
                row.update(losses=losses if index==len(results)-1 else {},
                           optimizer_update=update_due and index==len(results)-1,
                           last_optimizer_episode=last_optimizer_episode if index==len(results)-1 else (rows[-1]['last_optimizer_episode'] if rows else 0),
                           total_transitions=total_transitions,worker_timing=result['timing'],
                           worker_pid=result['worker_pid'],batch_performance=batch_perf)
                writer.record('metrics',episode,step,row);rows.append(row)
                writer.status(episode=episode,step=step,seed=row['episode_seed'],phase='recording',
                              completed_episodes=len(rows),total_transitions=total_transitions,
                              transitions_per_second=total_transitions/max(time.perf_counter()-started,1e-9))
                if mode=='train':
                    trained.append(row['episode_seed'])
                    if episode%c['checkpoint_every']==0 or episode==len(episode_seeds):
                        # Pending transitions are not model updates. A checkpoint
                        # at episode 50 can correctly contain the update at 48.
                        cutoff=last_optimizer_episode
                        seen=episode_seeds[:cutoff]
                        save_checkpoint(writer.path/'checkpoints'/f'episode_{episode:06d}.pt',
                                        dict(runtime=runtime,encoder=ENCODER,config=c,upstream=policy.provenance,
                                             budget_sha256=metadata['budget_sha256'],training_episode_seeds=seen,
                                             episode=episode,last_optimizer_episode=cutoff,
                                             total_transitions=total_transitions,
                                             policy=policy.checkpoint()))
                print(json.dumps(dict(episode=episode,seed=row['episode_seed'],official_team_return=row['official_team_return'],
                                      mean_cost=row['safety']['mean_cost'],budget_rate=rate,
                                      batch_transitions_per_second=batch_perf['transitions_per_second'])),flush=True)
            atomic_json(writer.path/'performance.json',dict(
                completed_episodes=len(rows),total_transitions=total_transitions,
                elapsed_seconds=time.perf_counter()-started,
                overall_transitions_per_second=total_transitions/max(time.perf_counter()-started,1e-9),
                last_batch=batch_perf,worker_pids=pool.worker_pids if pool is not None else []))
            start+=size
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
