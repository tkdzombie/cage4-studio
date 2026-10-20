"""Genuine five-agent independent actor-critic research BASELINE for CAGE 4.

This is a pedagogical on-policy REINFORCE-with-value-baseline prototype, NOT
PPO/IPPO, and NOT an official challenge result. The actor never reads the
CybORG true state. Each of the five blue policies receives only the official
BlueFlatWrapper observation and the legal action mask. Researchers alone may
inspect the independently serialized ground-truth network topology.
"""
from __future__ import annotations
import argparse
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

from .engine import create_env
from .runs import dump_json_atomic
from .topology import snapshot


def required_config(raw):
    out = {
        'seed': int(raw.get('seed', 7629)),
        'steps': int(raw.get('steps', 100)),
        'episodes': int(raw.get('episodes', 30)),
        'red': str(raw.get('red', 'FiniteStateRedAgent')),
        'lr': float(raw.get('lr', 0.0003)),
        'gamma': float(raw.get('gamma', 0.99)),
        'entropy_coef': float(raw.get('entropy_coef', 0.01)),
        'device': str(raw.get('device', 'cpu')),
        'snapshot_every': int(raw.get('snapshot_every', 10)),
        'checkpoint_every': int(raw.get('checkpoint_every', 5)),
    }
    from .engine import RED_NAMES
    if out['red'] not in RED_NAMES:
        raise ValueError('Unexpected red agent')
    if out['device'] not in ('cpu', 'mps'):
        raise ValueError('device must be cpu or mps')
    if not (0 <= out['seed'] < 2**32 and 10 <= out['steps'] <= 1000
            and 1 <= out['episodes'] <= 100000 and 0 < out['lr'] <= 0.1
            and 0 < out['gamma'] <= 1 and 0 <= out['entropy_coef'] <= 1
            and 1 <= out['snapshot_every'] <= 1000
            and 1 <= out['checkpoint_every'] <= 100000):
        raise ValueError('Invalid training parameter value')
    return out


def execute(run_dir: Path, config: dict):
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.distributions import Categorical
    from torch.utils.tensorboard import SummaryWriter
    from CybORG.Agents.Wrappers import BlueFlatWrapper

    config = required_config(config)
    torch.set_num_threads(min(8, os.cpu_count() or 4))
    if config['device'] == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable: choose CPU in the dashboard')
    dev = torch.device(config['device'])
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    random.seed(config['seed'])
    base = create_env(seed=config['seed'], steps=config['steps'], red=config['red'])
    env = BlueFlatWrapper(base, pad_spaces=True)
    obs, info = env.reset()
    agents = sorted(obs)
    if len(agents) != 5:
        raise RuntimeError(f'Expected five blue agents, got {agents}')

    class ActorCritic(nn.Module):
        def __init__(self, obs_dim, actions):
            super().__init__()
            self.shared = nn.Sequential(nn.Linear(obs_dim, 128), nn.Tanh(),
                                        nn.Linear(128, 128), nn.Tanh())
            self.actor = nn.Linear(128, actions)
            self.critic = nn.Linear(128, 1)

        def forward(self, features):
            h = self.shared(features)
            return self.actor(h), self.critic(h).squeeze(-1)

    policies, opts = {}, {}
    labels = {}
    for a in agents:
        dim = int(np.asarray(obs[a]).size)
        act = env.action_space(a)
        model = ActorCritic(dim, int(act.n)).to(dev)
        policies[a] = model
        opts[a] = torch.optim.Adam(model.parameters(), lr=config['lr'])
        labels[a] = list(map(str, env.action_labels(a)))

    writer = SummaryWriter(log_dir=str(run_dir / 'tensorboard'))
    dump_json_atomic(run_dir / 'policy_spaces.json', {
        a: {'observation_dim': int(np.asarray(obs[a]).size),
            'num_actions': int(env.action_space(a).n), 'action_labels': labels[a]}
        for a in agents
    })
    metric_file = (run_dir / 'metrics.jsonl').open('a', encoding='utf-8', buffering=1)
    event_file = (run_dir / 'actions.jsonl').open('a', encoding='utf-8', buffering=1)
    state = {'state': 'running', 'episode': 0, 'step': 0,
             'episodes_total': config['episodes'], 'started_at': time.time(),
             'pid': os.getpid(), 'device': config['device']}
    dump_json_atomic(run_dir / 'status.json', state)
    total_steps = 0
    try:
        for ep in range(1, config['episodes'] + 1):
            if (run_dir / 'stop.request').exists():
                state.update(state='stopped', stopped_at=time.time())
                dump_json_atomic(run_dir / 'status.json', state)
                return
            # Resets simulation through official wrapper; subnet dynamics are simulated.
            obs, info = env.reset()
            # Labels for absent/present hosts can change across episodes although
            # fixed action indices remain stable. Log the current episode labels.
            labels = {a: list(map(str, env.action_labels(a))) for a in agents}
            memories = {a: {'logp': [], 'value': [], 'entropy': [], 'reward': []}
                        for a in agents}
            episode_scores = {a: 0.0 for a in agents}
            episode_steps = 0
            snap = snapshot(base, episode=ep, step=0)
            dump_json_atomic(run_dir / 'topology_latest.json', snap)
            ep_snapshots = run_dir / 'snapshots' / f'episode_{ep:06d}'
            ep_snapshots.mkdir(parents=True, exist_ok=True)
            dump_json_atomic(ep_snapshots / 'step_0000.json', snap)
            for step in range(1, config['steps'] + 1):
                if (run_dir / 'stop.request').exists():
                    state.update(state='stopped', episode=ep, step=step - 1,
                                 stopped_at=time.time())
                    dump_json_atomic(run_dir / 'status.json', state)
                    return
                acts = {}
                action_labels = {}
                for a in agents:
                    if a not in obs:
                        continue
                    x = torch.as_tensor(np.asarray(obs[a], dtype=np.float32), device=dev).unsqueeze(0)
                    logits, val = policies[a](x)
                    legal = info.get(a, {}).get('action_mask', [])
                    if len(legal) != logits.shape[-1]:
                        raise RuntimeError(f'Action mask size mismatch for {a}: '
                                           f'{len(legal)} != {logits.shape[-1]}')
                    mask = torch.as_tensor(legal, dtype=torch.bool, device=dev).unsqueeze(0)
                    if not bool(mask.any().item()):
                        raise RuntimeError(f'No legal actions for {a}')
                    masked = logits.masked_fill(~mask, -1e9)
                    dist = Categorical(logits=masked)
                    action = dist.sample()
                    index = int(action.item())
                    acts[a] = index
                    action_labels[a] = labels[a][index]
                    memories[a]['logp'].append(dist.log_prob(action).squeeze(0))
                    memories[a]['value'].append(val.squeeze(0))
                    memories[a]['entropy'].append(dist.entropy().squeeze(0))
                obs, rewards, terminated, truncated, info = env.step(acts)
                total_steps += 1
                episode_steps = step
                for a in acts:
                    reward = float(rewards.get(a, 0.0))
                    episode_scores[a] += reward
                    memories[a]['reward'].append(reward)
                event_file.write(json.dumps({
                    'episode': ep, 'step': step,
                    'blue_actions': action_labels,
                    'blue_rewards': {a: float(rewards.get(a, 0.0)) for a in agents},
                    'red_actions': {a: str(base.get_last_action(a))
                                    for a in base.agents if a.startswith('red')},
                }, ensure_ascii=False) + '\n')
                if step == 1 or step % config['snapshot_every'] == 0 or step == config['steps']:
                    snap = snapshot(base, episode=ep, step=step)
                    dump_json_atomic(run_dir / 'topology_latest.json', snap)
                    # Persistent per-episode timeline; never overwrite previous episodes.
                    dump_json_atomic(ep_snapshots / f'step_{step:04d}.json', snap)
                    state.update(episode=ep, step=step, total_steps=total_steps,
                                 episode_running_reward=round(sum(episode_scores.values()), 3))
                    dump_json_atomic(run_dir / 'status.json', state)
                if not obs or all(terminated.get(a, False) or truncated.get(a, False)
                                  for a in agents):
                    break
            losses = {}
            for a in agents:
                mem = memories[a]
                if not mem['reward']:
                    continue
                g, rev = 0.0, []
                for r in reversed(mem['reward']):
                    g = r + config['gamma'] * g
                    rev.append(g)
                ret = torch.as_tensor(list(reversed(rev)), dtype=torch.float32, device=dev)
                logp = torch.stack(mem['logp'])
                values = torch.stack(mem['value'])
                entropy = torch.stack(mem['entropy'])
                advantage = ret - values.detach()
                actor_loss = -(logp * advantage).mean()
                critic_loss = 0.5 * (values - ret).square().mean()
                loss = actor_loss + critic_loss - config['entropy_coef'] * entropy.mean()
                opts[a].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policies[a].parameters(), max_norm=1.0)
                opts[a].step()
                losses[a] = round(float(loss.detach().cpu()), 5)
                writer.add_scalar(f'agent_return/{a}', episode_scores[a], ep)
                writer.add_scalar(f'loss/{a}', losses[a], ep)
            row = {'episode': ep, 'steps': episode_steps,
                   'team_return': round(sum(episode_scores.values()), 4),
                   'agents': {a: round(episode_scores[a], 4) for a in agents},
                   'losses': losses, 'total_steps': total_steps, 'time': time.time()}
            metric_file.write(json.dumps(row, ensure_ascii=False) + '\n')
            writer.add_scalar('team_return', row['team_return'], ep)
            writer.flush()
            state.update(episode=ep, step=episode_steps, last_episode_return=row['team_return'],
                         episode_running_reward=None)
            dump_json_atomic(run_dir / 'status.json', state)
            if ep % config['checkpoint_every'] == 0 or ep == config['episodes']:
                torch.save({'episode': ep, 'config': config,
                            'models': {a: policies[a].state_dict() for a in agents},
                            'optimizers': {a: opts[a].state_dict() for a in agents},
                            'spaces': {a: (int(np.asarray(obs[a]).size), int(env.action_space(a).n))
                                       for a in agents if a in obs}},
                           run_dir / 'checkpoints' / f'episode_{ep:05d}.pt')
        state.update(state='finished', finished_at=time.time())
        dump_json_atomic(run_dir / 'status.json', state)
    finally:
        event_file.close()
        metric_file.close()
        writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    args = parser.parse_args()
    path = args.run_dir.resolve()
    try:
        config = json.loads(args.config.read_text(encoding='utf-8'))
        execute(path, config)
    except Exception as exc:
        dump_json_atomic(path / 'status.json', {
            'state': 'error', 'error': str(exc), 'traceback': traceback.format_exc(),
            'finished_at': time.time(), 'pid': os.getpid(),
        })
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
