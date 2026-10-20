"""Minimal real CAGE 4 environment test; no fabricated data."""
from cage4studio.engine import create_env
from cage4studio.topology import snapshot
from CybORG.Agents.Wrappers import BlueFlatWrapper

env = create_env(seed=7629, steps=20)
env.reset()
env.step()
s = snapshot(env, episode=1, step=1)
assert len(s['hosts']) > 20, 'Expected a dynamically generated enterprise network'
assert len(s['links']) > 20, 'Expected actual interface data links'
assert s['provenance'].startswith('CybORG')
flat = BlueFlatWrapper(create_env(seed=7629, steps=20), pad_spaces=True)
obs, info = flat.reset()
assert len(obs) == 5
acts = {}
for a in obs:
    legal = info[a]['action_mask']
    acts[a] = next(i for i, is_valid in enumerate(legal) if is_valid)
obs2, rewards2, done2, trunc2, info2 = flat.step(acts)
assert all(a in rewards2 for a in acts)
assert all(a in info2 for a in acts)
print('WRAPPER STEP PASS: ', len(obs2), 'blue observations')
print('SMOKE TEST PASS:', len(s['hosts']), 'hosts', len(s['links']), 'interface links')
