"""Boundary checks for public-observation task diagnostics, not policy quality."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

path = Path(__file__).resolve().parents[2] / 'ops/probe_ot_task_validity.py'
spec = importlib.util.spec_from_file_location('task_probe', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
spec2 = importlib.util.spec_from_file_location('confirmation', path.with_name('ot_task_confirmation.py'))
confirmation = importlib.util.module_from_spec(spec2)
spec2.loader.exec_module(confirmation)


class RuleBoundaryTests(unittest.TestCase):
    def inputs(self, mask=(True, True, True), alert=True):
        schema = dict(head=['clock'], max_hosts=1,
                      hosts=[dict(slot=0, hostname='h', fields=['process_history', 'connection_history'])])
        # The object contains no simulator, raw state, OT field or cost.
        value = SimpleNamespace(observation=np.array([0., float(alert), 0.]),
                                labels=('Sleep', 'Restore h', 'BlockTrafficZone z <- y'),
                                mask=np.array(mask))
        return {'blue_agent_1': value}, {'blue_agent_1': schema}

    def test_reactive_uses_public_history(self):
        inputs, schemas = self.inputs()
        policy = module.PublicRule('alert_restore', schemas)
        self.assertEqual(policy.act(inputs), {'blue_agent_1': 1})
        inputs['blue_agent_1'].observation[:] = 0
        self.assertEqual(policy.act(inputs), {'blue_agent_1': 0})

    def test_busy_mask_cannot_issue_ignored_restore(self):
        inputs, schemas = self.inputs((True, False, False))
        for name in module.RULES:
            self.assertEqual(module.PublicRule(name, schemas).act(inputs), {'blue_agent_1': 0})

    def test_barrier_attempted_once_then_reactive(self):
        inputs, schemas = self.inputs()
        policy = module.PublicRule('mission_barriers', schemas)
        self.assertEqual(policy.act(inputs), {'blue_agent_1': 2})
        self.assertEqual(policy.act(inputs), {'blue_agent_1': 1})

    def test_zero_events_need_enough_independent_episodes(self):
        self.assertGreater(confirmation.upper_bound(0, 58), .05)
        self.assertLess(confirmation.upper_bound(0, 59), .05)
        self.assertAlmostEqual(confirmation.upper_bound(0, 60), 1-.05**(1/60))

    def test_one_event_in_sixty_does_not_certify_five_percent(self):
        self.assertGreater(confirmation.upper_bound(1, 60), .05)
        self.assertEqual(confirmation.upper_bound(60, 60), 1.)

    def test_invalid_risk_evidence_is_rejected(self):
        for k,n in ((1,0),(-1,60),(61,60)):
            with self.assertRaises(ValueError):
                confirmation.upper_bound(k,n)


if __name__ == '__main__':
    unittest.main()
