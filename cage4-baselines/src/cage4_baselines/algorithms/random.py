import numpy as np
from .registry import register

@register('random')
class RandomPolicy:
    def __init__(self, inputs, config):
        self.rng = np.random.default_rng(config['seed'])
    def act(self, inputs, training=False):
        return {a:int(self.rng.choice(np.flatnonzero(x.mask))) for a,x in inputs.items()}
    def observe(self, rewards):
        pass
    def update(self):
        return {}
    def checkpoint(self):
        return {}
    def load(self, data):
        pass
