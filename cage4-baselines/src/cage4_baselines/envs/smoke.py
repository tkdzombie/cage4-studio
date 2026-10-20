"""Run a complete real simulator episode using only the public adapter API."""
import numpy as np
from .adapter import Cage4Adapter
from ..algorithms import create


def smoke(seed=7629, steps=20):
    env = Cage4Adapter(seed=seed, steps=steps)
    inputs = env.reset()
    spaces = {a: {'observation_dim': len(x.observation), 'actions': len(x.mask),
                  'legal_actions': int(x.mask.sum())} for a, x in inputs.items()}
    policy = create('random', inputs, {'seed': seed, 'plugins': []})
    returns = {a: 0.0 for a in inputs}
    executed = 0
    for _ in range(steps):
        if not all(np.isfinite(x.observation).all() for x in inputs.values()):
            raise RuntimeError('Non-finite Blue observation')
        actions = policy.act(inputs)
        inputs, rewards, terminated, truncated = env.step(actions)
        executed += 1
        for agent in returns:
            returns[agent] += rewards.get(agent, 0.0)
        if not inputs or all(terminated.get(a, False) or truncated.get(a, False) for a in returns):
            break
    else:
        raise RuntimeError('Environment did not finish within the configured horizon')
    return {'status': 'PASS', 'seed': seed, 'steps_executed': executed,
            'spaces': spaces, 'returns': returns, 'selected_actions': 'official mask legal'}
