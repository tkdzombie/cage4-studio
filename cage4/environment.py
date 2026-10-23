"""Author marl-1policy PPO environment with the same explicit semantic repairs."""
import os, sys
from pathlib import Path
import numpy as np
ROOT=Path(os.environ.get('CAGE4_ROOT',Path(__file__).resolve().parents[1]))
UPSTREAM=ROOT/'Hierarchical-MARL'/'marl-1policy'
sys.path.insert(0,str(UPSTREAM))
import BlueFlatWrapper_CC4 as flat_source
from EnterpriseMAE_CC4 import EnterpriseMAE
from action_mask_model_CC4 import TorchActionMaskModel
from ray.rllib.models import ModelCatalog
ModelCatalog.register_custom_model('my_model',TorchActionMaskModel)
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from gymnasium import spaces
from CybORG import CybORG
from CybORG.Agents import SleepAgent,EnterpriseGreenAgent,FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

class RepairedFlat(EnterpriseMAE):
    def action_mask(self,agent): return list(super().action_mask(agent))
    def erase_history_if_recovered(self,agent,observation,subnet,hosts):
        if getattr(observation.get('action'),'name','')=='Remove':return
        return super().erase_history_if_recovered(agent,observation,subnet,hosts)
    def observation_change(self,agent,observation):
        busy=getattr(observation.get('success'),'name','')=='IN_PROGRESS'
        source=dict(observation,success=type(observation['success']).UNKNOWN) if busy else observation
        out=super().observation_change(agent,source)
        legal=np.asarray(self.action_mask(agent),dtype=bool)
        labels=self.action_labels(agent)
        legal &= np.asarray([not x.startswith(('[Invalid]','[Padding]')) for x in labels])
        mask=np.asarray(out['action_mask'],dtype=bool)&legal
        if busy:mask[:]=False
        if busy or not mask.any():mask[labels.index('Sleep')]=True
        out['action_mask']=mask.astype(np.int64)
        return out

class PrimitiveEnv(MultiAgentEnv):
    def __init__(self,config=None):
        super().__init__();config=config or {}
        self.protocol=config.get('protocol','flat-history-correct')
        flat_source.USE_BLOCK = self.protocol=='flat-full-actions-history-correct'
        self.base_seed=int(config.get('seed',321))+int(getattr(config,'worker_index',0))*100003
        self.episode_index=0
        g=EnterpriseScenarioGenerator(blue_agent_class=SleepAgent,green_agent_class=EnterpriseGreenAgent,red_agent_class=FiniteStateRedAgent,steps=500)
        self.inner=RepairedFlat(CybORG(scenario_generator=g,seed=self.base_seed))
        self._agent_ids={f'blue_agent_{i}' for i in range(5)}
        self.observation_space=spaces.Dict({a:spaces.Dict({'observations':self.inner.observation_space(a)['observations'],'action_mask':spaces.MultiBinary(self.inner.action_space(a).n)}) for a in self._agent_ids})
        self.action_space=spaces.Dict({a:self.inner.action_space(a) for a in self._agent_ids})
        self._spaces_in_preferred_format=True
        self.team_return=0.0;self.ticks=0
    def reset(self,*,seed=None,options=None):
        if seed is None:seed=self.base_seed+self.episode_index
        self.episode_index+=1;self.team_return=0.0;self.ticks=0
        obs,info=self.inner.reset(seed=seed)
        self.current=obs
        return obs,info
    def step(self,actions):
        assert set(actions)==self._agent_ids
        assert all(self.current[a]['action_mask'][int(k)] for a,k in actions.items())
        obs,rewards,_,_,info=self.inner.step(actions)
        self.ticks+=1
        values=list(rewards.values());assert np.allclose(values,values[0])
        self.team_return+=float(values[0]);done=bool(self.inner.env.environment_controller.determine_done())
        terminated={a:done for a in self._agent_ids};terminated['__all__']=done
        truncated={a:False for a in self._agent_ids};truncated['__all__']=False
        for a in obs:info[a]={'team_return':self.team_return,'ticks':self.ticks,'action_mask':obs[a]['action_mask']}
        self.current=obs
        return obs,rewards,terminated,truncated,info

# Old saved RLlib configs refer to this module name. Keep model loading compatible.
sys.modules.setdefault("primitive_env", sys.modules[__name__])
