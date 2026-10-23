"""Versioned repairs around pristine H-MARL source. Simulator/reward unchanged."""
import os
import sys
from pathlib import Path
import numpy as np

ROOT = Path(os.environ.get('CAGE4_ROOT', Path.home() / 'cage4-research'))
UPSTREAM = ROOT / 'Hierarchical-MARL' / 'h-marl-4policy'
sys.path.insert(0, str(UPSTREAM / 'master'))
from ray.rllib.policy.policy import Policy
from ray.rllib.models import ModelCatalog
from action_mask_model_CC4 import TorchActionMaskModel
ModelCatalog.register_custom_model('my_model', TorchActionMaskModel)
import BlueFlatWrapper_CC4 as upstream_flat
from EnterpriseMAE_CC4 import EnterpriseMAE
from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator
from gymnasium import spaces

_CACHE = {}
class FrozenAgent:
    def __init__(self, name):
        if name not in _CACHE:
            folder = 'master/iter_49' if 'master' in name else 'sub/iter_30'
            _CACHE[name] = Policy.from_checkpoint(str(UPSTREAM / 'saved_policies' / folder / 'policies' / name))
        self.policy = _CACHE[name]
    def get_action(self, observation, action_space=None):
        self.last_action = int(self.policy.compute_single_action(observation, explore=True)[0])
        return self.last_action
upstream_flat.HMARLBlueAgent = FrozenAgent

class InnerMaster(EnterpriseMAE):
    """author-state-v1 reference; no red session/infection truth added.

    Original event/block/decoy features are retained and explicitly recorded.
    Repairs: fresh mask, fixed horizon terminal, team return not fivefold sum.
    """
    def __init__(self, config=None):
        config = config or {}
        self.protocol = config.get('protocol', 'repaired')
        self.base_seed = int(config.get('seed', 321)) + int(getattr(config, 'worker_index', 0)) * 100003
        self.episode_index = 0
        gen = EnterpriseScenarioGenerator(blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent, steps=500)
        super().__init__(CybORG(scenario_generator=gen, seed=self.base_seed))
        self._agent_ids = {f'blue_agent_{i}_master' for i in range(5)}
        self._spaces_in_preferred_format = False
        self.team_return = 0.0
        self.ticks = 0
        self.mask_mismatches = 0

    def action_mask(self, agent):
        mask = super().action_mask(agent)
        # The upstream encoder mutates this list for mission traffic filtering.
        # Keep simulator legality immutable in the repaired-v2 protocol.
        return list(mask) if self.protocol in ('valid-actions', 'valid-actions-info', 'history-correct') else mask

    def erase_history_if_recovered(self, agent, observation, subnet, hosts):
        if self.protocol == 'history-correct' and getattr(observation.get('action'),'name','') == 'Remove':
            # TRUE means the attempt ran, not that privileged persistence or files disappeared.
            return
        return super().erase_history_if_recovered(agent,observation,subnet,hosts)

    def observation_change(self, agent, observation):
        busy = getattr(observation.get('success'), 'name', '') == 'IN_PROGRESS'
        source_observation = observation
        if busy and self.protocol in ('valid-actions-info','history-correct'):
            # Keep real observed features for the value network. No extra action
            # becomes available: all three subpolicies still receive Sleep only.
            source_observation = dict(observation, success=type(observation['success']).UNKNOWN)
        values = list(super().observation_change(agent, source_observation))
        if busy and self.protocol in ('valid-actions-info','history-correct'):
            sleep = self.action_labels(agent).index('Sleep')
            for j in (4,5,6):
                values[j] = np.zeros_like(values[j]); values[j][sleep] = 1
            self.high_actionmask[agent] = np.eye(3,dtype=np.int64)[self.prev_action[agent]]
        if self.protocol in ('valid-actions', 'valid-actions-info', 'history-correct'):
            legal = np.asarray(self.action_mask(agent), dtype=bool)
            labels = self.action_labels(agent)
            legal &= np.asarray([not x.startswith(('[Invalid]', '[Padding]')) for x in labels])
            for j in (4,5,6):
                values[j] = (np.asarray(values[j], dtype=bool) & legal).astype(np.int64)
                if not values[j].any(): values[j][labels.index('Sleep')] = 1
        return tuple(values)

    def master_space(self, a):
        return spaces.Dict({'action_mask': spaces.MultiBinary(3),
            'observations': self._observation_space[a]['observations'], 'id': spaces.Discrete(2)})

    def reset(self, *, seed=None, options=None, **kwargs):
        if seed is None:
            seed = self.base_seed + self.episode_index
        self.episode_index += 1
        self.team_return = 0.0
        self.ticks = 0
        self.mask_mismatches = 0
        obs, info = super().reset(seed=seed)
        return obs, info

    def step(self, action_dict):
        obs, rewards, terminated, truncated, info = super().step(action_dict)
        self.ticks += 1
        # Five agents receive identical team reward; report ONE team return.
        vals = list(rewards.values())
        assert np.allclose(vals, vals[0]), rewards
        self.team_return += float(vals[0])
        done = bool(self.env.environment_controller.determine_done())
        if self.protocol != 'original':
            for agent, o in obs.items():
                base = agent[:12]
                fresh = self.high_actionmask[base].copy()
                self.mask_mismatches += int(not np.array_equal(o['action_mask'], fresh))
                o['action_mask'] = fresh
                info[agent]['action_mask'] = fresh.copy()
            # 500-step task is a finite-horizon objective, not an external cutoff.
            terminated = {a: done for a in self._agent_ids}
            terminated['__all__'] = done
            truncated = {a: False for a in self._agent_ids}
            truncated['__all__'] = False
        for agent in obs:
            info[agent]['team_return'] = self.team_return
            info[agent]['ticks'] = self.ticks
            info[agent]['mask_mismatches'] = self.mask_mismatches
        return obs, rewards, terminated, truncated, info


from ray.rllib.env.multi_agent_env import MultiAgentEnv
class MasterEnv(MultiAgentEnv):
    """Composition keeps upstream callable spaces out of RLlib serialization."""
    def __init__(self, config=None):
        super().__init__()
        self.inner = InnerMaster(config)
        self._agent_ids = self.inner._agent_ids
        self.observation_space = spaces.Dict({a:self.inner.master_space(a[:12]) for a in self._agent_ids})
        self.action_space = spaces.Dict({a:spaces.Discrete(3) for a in self._agent_ids})
        self._spaces_in_preferred_format = True
    def master_space(self, a): return self.inner.master_space(a)
    def reset(self, *, seed=None, options=None): return self.inner.reset(seed=seed,options=options)
    def step(self, actions): return self.inner.step(actions)
    @property
    def team_return(self): return self.inner.team_return
    @property
    def ticks(self): return self.inner.ticks
    @property
    def mask_mismatches(self): return self.inner.mask_mismatches
    @property
    def high_actionmask(self): return self.inner.high_actionmask
