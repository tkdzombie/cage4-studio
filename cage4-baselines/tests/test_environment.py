"""Real CybORG contract tests; skipped in lightweight CI without the simulator."""
import importlib.util
import unittest
import numpy as np
from cage4_baselines.envs.adapter import Cage4Adapter
from cage4_baselines.algorithms import create


@unittest.skipUnless(importlib.util.find_spec('CybORG'), 'real CybORG runtime required')
class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.env = Cage4Adapter(seed=42, steps=10)

    def test_public_reset_step_tracks_state_automatically(self):
        inputs = self.env.reset()
        self.assertEqual(len(inputs), 5)
        policy = create('random', inputs, {'seed':42, 'plugins':[]})
        for _ in range(2):
            actions = {a:np.int64(i) for a,i in policy.act(inputs).items()}
            inputs, rewards, terminated, truncated = self.env.step(actions)
            self.assertIs(inputs, self.env.current)
            self.assertEqual(set(rewards), set(actions))
            self.assertTrue(all(len(x.observation)==self.env.observation_schema()[a]["dimension"] for a,x in inputs.items()))

    def test_rejects_missing_agent_and_masked_action_before_execution(self):
        inputs = self.env.reset()
        policy = create('random', inputs, {'seed':42, 'plugins':[]})
        actions = policy.act(inputs)
        missing = dict(actions);missing.pop(next(iter(actions)))
        with self.assertRaises(ValueError):self.env.step(missing)
        agent = next(a for a,x in inputs.items() if not x.mask.all())
        invalid = dict(actions);invalid[agent] = int(np.flatnonzero(~inputs[agent].mask)[0])
        with self.assertRaises(ValueError):self.env.step(invalid)
        boolean = dict(actions);boolean[agent] = True
        with self.assertRaises(ValueError):self.env.step(boolean)
        self.assertIs(inputs, self.env.current)

    def test_reset_required_before_and_after_episode(self):
        with self.assertRaises(RuntimeError):self.env.step({})
        inputs = self.env.reset()
        policy = create('sleep', inputs, {'seed':42, 'plugins':[]})
        for _ in range(10):
            inputs,_,done,truncated = self.env.step(policy.act(inputs))
            if not inputs or all(done.get(a,False) or truncated.get(a,False) for a in done):break
        else:self.fail('Episode did not complete')
        with self.assertRaises(RuntimeError):self.env.step({})
        self.assertEqual(len(self.env.reset()),5)

    def test_busy_actions_only_allow_sleep_and_schema_stable_across_seeds(self):
        inputs = self.env.reset()
        dimensions = {a:x.observation.shape for a,x in inputs.items()}
        for a,x in self.env.reset(43).items(): self.assertEqual(x.observation.shape, dimensions[a])
        from types import SimpleNamespace
        raw = {a:{'success':SimpleNamespace(name='IN_PROGRESS')} for a in inputs}
        info = {a:{'action_mask':self.env.env.action_mask(a)} for a in inputs}
        for x in self.env._inputs(raw,info).values():
            self.assertEqual(int(x.mask.sum()),1)
            self.assertEqual(x.labels[int(np.flatnonzero(x.mask)[0])],'Sleep')
