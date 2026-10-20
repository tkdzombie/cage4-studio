"""Focused real-runtime acceptance for the constrained experiment campaign."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

REAL=bool(importlib.util.find_spec('torch') and importlib.util.find_spec('CybORG'))


@unittest.skipUnless(REAL,'real CybORG/Torch runtime required')
class ConstrainedTests(unittest.TestCase):
    def test_both_safe_trainers_and_explicit_evaluation_modes(self):
        import torch
        from cage4_baselines.paper.__main__ import run
        root=Path(__file__).resolve().parents[1]
        budget=root/'configs/safety/availability-target-v2.json'
        for name in ('macpo','mappo_lagrangian'):
            c=dict(algorithm=name,seed=101,episodes=2,steps=20,reward_scale=.01,rollout_episodes=2,
                   num_envs=2,torch_threads=1,checkpoint_every=50,record_replay=False,deterministic_eval=False,
                   hyperparameters=dict(hidden_size=32,layer_N=1,ppo_epoch=2))
            with self.subTest(algorithm=name),tempfile.TemporaryDirectory() as t:
                train=run(c,Path(t)/'train',Path.home()/'cage4-upstreams',budget,smoke=True)
                rows=[json.loads(p.read_text()) for p in sorted((train/'metrics').rglob('*.json'))]
                for values in rows[-1]['losses'].values():
                    self.assertAlmostEqual(values['budget_episode'],.95)
                    self.assertGreaterEqual(values['actual_policy_kl'],-1e-6)
                    self.assertGreaterEqual(values['cost_value_mse_after_update'],0)
                    if name=='macpo':
                        self.assertLessEqual(values['actual_policy_kl'],.01001)
                        self.assertGreaterEqual(values['cpo_cost_dual'],0)
                        self.assertNotIn('lam',values)
                        self.assertNotIn('nu',values)
                cp=train/'checkpoints/episode_000002.pt'
                payload=torch.load(cp,weights_only=True)
                for mode in ('stochastic','greedy'):
                    evaluated=run(c,Path(t)/'eval',Path.home()/'cage4-upstreams',budget,mode='evaluate',
                                  checkpoint=cp,seeds=[54011,54012],smoke=True,evaluation_action_mode=mode)
                    self.assertEqual(json.loads((evaluated/'evaluation.json').read_text())['action_mode'],mode)
                    self.assertEqual(json.loads((evaluated/'config.json').read_text()),payload['config'])

    def test_lagrangian_multiplier_responds_to_budget_in_correct_direction(self):
        import torch
        from cage4_baselines.envs.adapter import Cage4Adapter
        from cage4_baselines.paper.backend import AuthorBaseline
        torch.set_num_threads(1)
        c=dict(algorithm='mappo_lagrangian',steps=20,hyperparameters=dict(hidden_size=32,ppo_epoch=1,
                                                                      lagrangian_coef_rate=.01))
        env=Cage4Adapter(seed=101,steps=20,include_graph=False);inputs=env.reset()
        values=[]
        for rate in (.05,.30):
            torch.manual_seed(42)
            policy=AuthorBaseline(inputs,c,Path.home()/'cage4-upstreams',rate)
            policy.begin_episode()
            for step in range(2):
                policy.act(inputs,training=True)
                policy.observe({a:float(step+1) for a in inputs},.2,terminal=step==1)
            metrics=policy.update(.01)
            values.append([metrics[a]['lagrange_multiplier'] for a in sorted(inputs)])
        self.assertTrue(all(low>high for low,high in zip(*values)))
