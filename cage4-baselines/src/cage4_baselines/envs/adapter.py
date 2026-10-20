"""Official fixed action wrapper + observation-only features; no BlueFlatWrapper."""
import numpy as np
from .factory import BLUE_NAMES, create_env
from .observations import encode
from ..core.types import PolicyInput

class Cage4Adapter:
    def __init__(self, seed, steps, red, obs_dim=512):
        from CybORG.Agents.Wrappers import BlueFixedActionWrapper
        self.base = create_env(seed=seed, steps=steps, red=red)
        self.env = BlueFixedActionWrapper(self.base, pad_spaces=True)
        self.obs_dim = obs_dim
    def _inputs(self, raw, info):
        self.raw = raw
        return {a:PolicyInput(encode(raw[a],self.obs_dim),
                              np.asarray(info[a]['action_mask'],dtype=bool).copy(),
                              tuple(map(str,self.env.action_labels(a)))) for a in raw}
    def reset(self, seed):
        raw,info = self.env.reset(seed=seed)
        if set(raw) != set(BLUE_NAMES):
            raise RuntimeError('Expected exactly five Blue agents')
        return self._inputs(raw,info)
    def step(self, actions):
        for a,i in actions.items():
            if not isinstance(i,int) or i < 0 or i >= len(self.current[a].mask) or not self.current[a].mask[i]:
                raise ValueError(f'Illegal action index: {a}/{i}')
        raw,rew,terminated,truncated,info = self.env.step(actions)
        return self._inputs(raw,info), {a:float(v) for a,v in rew.items()}, terminated,truncated
    def inputs(self, values):
        self.current = values
        return values
