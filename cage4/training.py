"""Fixed-budget PPO training, checkpoint lifecycle and reproducible run records."""
import hashlib
import json
import os
import pickle
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import ray
import torch
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

from algorithms.ppo import ScaledRewardPPO
from cage4.config import ROOT
from cage4.environment import PrimitiveEnv, UPSTREAM


class Metrics(DefaultCallbacks):
    def on_episode_end(self, *, episode, **kwargs):
        info = episode.last_info_for('blue_agent_0') or {}
        for key in ('team_return', 'ticks'):
            if key in info:
                episode.custom_metrics[key] = info[key]


def policy_mapping(agent_id, *args, **kwargs):
    return 'Agent' + agent_id.rsplit('_', 1)[1]


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=float))
    temporary.replace(path)


def train(settings):
    torch.set_num_threads(1)
    output = settings.output
    output.mkdir(parents=True, exist_ok=False)
    config = settings.to_dict()
    write_json(output / 'config.json', config)
    sources = [ROOT / 'run.py'] + sorted((ROOT/'cage4').glob('*.py')) + sorted((ROOT/'algorithms').glob('*.py'))
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        target = output / 'source' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = hashlib.sha256(target.read_bytes()).hexdigest()
    write_json(output/'provenance.json', {'source_sha256': hashes, 'ray': ray.__version__, 'torch': torch.__version__})
    status_path = output / 'status.json'
    write_json(status_path, {'phase': 'initializing', 'pid': os.getpid(), 'iterations': 0})
    if settings.algorithm == 'my_algorithm':
        from algorithms.my_algorithm import MyAlgorithmPolicy
        policy_class = MyAlgorithmPolicy
    else:
        policy_class = ScaledRewardPPO
    env = PrimitiveEnv({'seed': settings.seed, 'protocol': settings.protocol})
    specs = {f'Agent{i}': PolicySpec(policy_class, env.observation_space[f'blue_agent_{i}'],
             env.action_space[f'blue_agent_{i}'], {'learner_reward_scale': settings.reward_scale}) for i in range(5)}
    config_builder = (PPOConfig().framework('torch')
        .environment(PrimitiveEnv, env_config={'seed': settings.seed, 'protocol': settings.protocol}, disable_env_checking=True)
        .experimental(_disable_preprocessor_api=True)
        .rollouts(num_rollout_workers=settings.workers, rollout_fragment_length='auto', batch_mode='complete_episodes')
        .resources(num_gpus=0, num_cpus_per_worker=1)
        .training(model={'custom_model': 'my_model', 'fcnet_hiddens': [256,256]},
            train_batch_size=settings.batch, sgd_minibatch_size=settings.minibatch,
            num_sgd_iter=settings.epochs, lr=settings.lr, gamma=.99, lambda_=settings.gae_lambda,
            entropy_coeff=.001, vf_clip_param=10000.0, grad_clip=.5)
        .multi_agent(policies=specs, policy_mapping_fn=policy_mapping, policies_to_train=list(specs))
        .reporting(metrics_num_episodes_for_smoothing=1).callbacks(Metrics)
        .debugging(seed=settings.seed, log_level='ERROR'))
    algo = None
    requested = False
    def request_stop(*_):
        nonlocal requested
        requested = True
    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    best = -float('inf')
    rows = []
    started = time.time()
    reason = 'budget'
    try:
        ray.init(include_dashboard=False, num_cpus=settings.workers+2, log_to_driver=False)
        algo = config_builder.build()
        if settings.initialization == 'author':
            weights = {}
            for name in specs:
                with (UPSTREAM/'models/train_marl/iter_199/policies'/name/'policy_state.pkl').open('rb') as handle:
                    weights[name] = pickle.load(handle)['weights']
                for key in weights[name]:
                    if key.startswith('internal_model._value_branch.'):
                        weights[name][key] *= settings.reward_scale
            algo.set_weights(weights)
            algo.workers.sync_weights()
        algo.save(str(output/'initial'))
        for iteration in range(1, settings.iterations+1):
            if requested or (output/'STOP').exists():
                reason = 'requested'
                break
            result = algo.train()
            custom = result.get('custom_metrics', {})
            score = custom.get('team_return_mean', result['episode_reward_mean']/5)
            learners = result.get('info', {}).get('learner', {})
            row = {'iteration': iteration, 'seconds': time.time()-started,
                   'env_steps': result['timesteps_total'], 'return': score,
                   'episode_count': result.get('episodes_this_iter'),
                   'min_return': custom.get('team_return_min'), 'max_return': custom.get('team_return_max'),
                   'learner': {key: value.get('learner_stats', {}) for key,value in learners.items()}}
            with (output/'metrics.jsonl').open('a') as handle:
                handle.write(json.dumps(row, default=float)+'\n')
            rows.append(row)
            print(json.dumps({k:v for k,v in row.items() if k != 'learner'}), flush=True)
            finite = np.isfinite(score) and all(np.isfinite(value) for stats in row['learner'].values()
                       for value in stats.values() if isinstance(value, (int,float)))
            if not finite:
                reason = 'nonfinite_learner'
                break
            if score > best:
                best = score
                algo.save(str(output/'best'))
            if iteration % settings.checkpoint_every == 0:
                algo.save(str(output/'latest'))
            write_json(status_path, {'phase': 'training', 'pid': os.getpid(), 'iterations': iteration,
                       'env_steps': row['env_steps'], 'last_return': score, 'updated_at': time.time()})
        algo.save(str(output/'final'))
        summary = {'status': 'stopped', 'reason': reason, 'iterations': len(rows),
                   'env_steps': rows[-1]['env_steps'] if rows else 0,
                   'best_training_return': best if rows else None,
                   'last_return': rows[-1]['return'] if rows else None}
        write_json(output/'summary.json', summary)
        write_json(status_path, {'phase': 'evaluating' if reason=='budget' else 'stopped', **summary})
    except BaseException as error:
        write_json(status_path, {'phase': 'failed', 'error': repr(error), 'pid': os.getpid()})
        raise
    finally:
        if algo is not None:
            algo.stop()
        ray.shutdown()
    if reason == 'budget':
        from cage4.evaluation import evaluate
        evaluate(output, settings.evaluation_seeds, ['initial', 'final'])
        write_json(status_path, {'phase': 'complete', **summary})
    return summary
