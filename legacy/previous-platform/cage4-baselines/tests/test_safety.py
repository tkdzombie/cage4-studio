import unittest
import json
from pathlib import Path
import tempfile
from cage4_baselines.safety.costs import Workload, aggregate, summarize


class AvailabilityTests(unittest.TestCase):
    def test_new_targets_and_historical_budget_remain_distinct(self):
        from cage4_baselines.safety.budgets import read_budget
        root = Path(__file__).resolve().parents[1]/'configs/safety'
        old, rate = read_budget(root/'availability-v1.json',500)
        self.assertAlmostEqual(rate,.2474800289213457)
        path = root/'availability-target-v2.json'
        target, rate = read_budget(path,500)
        self.assertEqual(rate,.05)
        self.assertFalse(target['validation_mean_feasible'])
        self.assertEqual(target['feasibility'],'not_established')
        self.assertEqual(read_budget(path,500,level='strict')[1],.03)
        self.assertEqual(read_budget(path,500,level='loose')[1],.10)
        with self.assertRaises(ValueError): read_budget(path,20)
        with tempfile.TemporaryDirectory() as t:
            bad = Path(t)/'bad.json'
            for rates in (dict(strict_rate=.2,main_rate=.15,loose_rate=.1),
                          dict(strict_rate=.1,main_rate=float('nan'),loose_rate=.2),
                          dict(strict_rate=.1,main_rate=True,loose_rate=.2)):
                value = dict(target,budgets=rates)
                bad.write_text(json.dumps(value))
                with self.assertRaises(ValueError): read_budget(bad,500)

    def test_mission_weight_and_failure_denominator(self):
        tasks = [Workload(10, True, 'LWF', 'critical'), Workload(1, False, 'ASF', 'hq'),
                 Workload(0, True, 'ASF', 'disabled')]
        row = aggregate(tasks)
        self.assertAlmostEqual(row['cost'], 10/11)
        self.assertEqual(row['eligible_tasks'], 2)
        self.assertEqual(row['zero_weight_tasks'], 1)
        self.assertEqual(row['failed_weight'], 10)

    def test_empty_step_and_different_episode_aggregates(self):
        empty = aggregate([])
        one = aggregate([Workload(1, True, 'LWF', 'a')])
        many = aggregate([Workload(9, False, 'LWF', 'a')])
        s = summarize([one, many, empty])
        self.assertAlmostEqual(s['mean_cost'], 1/3)
        self.assertAlmostEqual(s['request_weighted_failure_rate'], .1)
        self.assertEqual(s['no_task_steps'], 1)
        self.assertEqual(s['episode_cost'], 1)

    def test_invalid_weights_do_not_hide_in_cost(self):
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError): aggregate([Workload(value, False, 'LWF', 'a')])

    def test_budget_rule_and_sensitivity_order(self):
        try:
            from cage4_baselines.safety.calibrate import propose, CALIBRATION_SEEDS, VALIDATION_SEEDS
        except ImportError: self.skipTest('numpy required')
        b = propose([.2]*20)
        self.assertAlmostEqual(b['main_rate'], .2)
        self.assertLessEqual(b['strict_rate'], b['main_rate'])
        self.assertLessEqual(b['main_rate'], b['loose_rate'])
        self.assertFalse(set(CALIBRATION_SEEDS) & set(VALIDATION_SEEDS))
        self.assertEqual(propose([0.]*20)['main_rate'], .001)
        with self.assertRaises(ValueError): propose([.1]*19)
        with self.assertRaises(ValueError): propose([1.1]*20)


class CategoricalPortTests(unittest.TestCase):
    def test_exact_kl_and_fisher_curvature(self):
        try:
            import torch
            from cage4_baselines.paper.discrete_macpo import categorical_kl, conjugate_gradient
        except ImportError: self.skipTest('torch optional in lightweight CI')
        from types import SimpleNamespace
        class Output(torch.nn.Module):
            def __init__(self, probabilities):
                super().__init__()
                self.logits = torch.nn.Parameter(torch.tensor(probabilities).log())
            def forward(self, x, mask):
                return torch.distributions.Categorical(logits=self.logits.expand(len(x), -1).masked_fill(mask == 0, -1e10))
        def actor(p):
            return SimpleNamespace(tpdv=dict(dtype=torch.float32, device='cpu'),
                base=torch.nn.Identity(), act=SimpleNamespace(action_out=Output(p)))
        old = actor([.2, .8]); new = actor([.5, .5])
        obs = [[1.]]; mask = [[1., 1.]]
        kl = categorical_kl(new, old, obs, mask).sum()
        expected = .2*torch.log(torch.tensor(.4)) + .8*torch.log(torch.tensor(1.6))
        self.assertAlmostEqual(float(kl), float(expected), places=6)
        same = categorical_kl(new, new, obs, mask).sum()
        p = new.act.action_out.logits
        grad = torch.autograd.grad(same, p, create_graph=True)[0]
        direction = torch.tensor([1., -1.])
        hvp = torch.autograd.grad((grad*direction).sum(), p)[0]
        self.assertGreater(float((hvp*direction).sum()), 0)
        self.assertTrue(torch.equal(conjugate_gradient(lambda x: x, torch.zeros(2)), torch.zeros(2)))
        result = conjugate_gradient(lambda x: 2*x, torch.tensor([2., 4.]))
        self.assertTrue(torch.allclose(result, torch.tensor([1., 2.])))
