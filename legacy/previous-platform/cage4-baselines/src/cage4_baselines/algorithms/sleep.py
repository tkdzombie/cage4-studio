from .registry import register
from .random import RandomPolicy

@register('sleep')
class SleepPolicy(RandomPolicy):
    def act(self, inputs, training=False):
        result = {}
        for a,x in inputs.items():
            candidates = [i for i,v in enumerate(x.labels) if x.mask[i] and v.startswith('Sleep')]
            if not candidates:
                raise RuntimeError('Official Sleep action missing')
            result[a] = candidates[0]
        return result
