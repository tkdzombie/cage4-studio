import copy
import importlib.util
from pathlib import Path
import unittest

RUNNER=Path(__file__).resolve().parents[2]/'ops/ot_continuity_baselines.py'
spec=importlib.util.spec_from_file_location('ot_campaign',RUNNER)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class MechanismPlanTests(unittest.TestCase):
    def plan(self):
        return dict(schema='cage4.ot-mechanism-candidates/1',training_seed=101,episodes=256,batch=16,evaluation_seeds=list(range(54001,54021)),max_candidates=2,max_seconds_per_task=21600,minimum_free_gib=30,candidates={'mappo_lagrangian':dict(hyperparameters={'lamda_lagr':100.},question='Cost gradient weight'),'macpo':dict(hyperparameters={'fraction_coef':1.},question='Step cap')})
    def test_exact_bounded_plan(self):
        p=self.plan();self.assertEqual(module.validate_mechanism_plan(p),p)
    def test_reject_cost_or_budget_override(self):
        p=self.plan();p['candidates']['macpo']['hyperparameters']['event_budget']=.1
        with self.assertRaises(ValueError):module.validate_mechanism_plan(p)
    def test_reject_extra_candidate(self):
        p=self.plan();p['candidates']['other']=copy.deepcopy(p['candidates']['macpo'])
        with self.assertRaises(ValueError):module.validate_mechanism_plan(p)
    def test_reject_unbounded_training(self):
        p=self.plan();p['episodes']=2000
        with self.assertRaises(ValueError):module.validate_mechanism_plan(p)
    def test_reject_final_scenes(self):
        p=self.plan();p['evaluation_seeds']=list(range(53001,53021))
        with self.assertRaises(ValueError):module.validate_mechanism_plan(p)
    def test_reject_boolean_hyperparameter(self):
        p=self.plan();p['candidates']['macpo']['hyperparameters']['fraction_coef']=True
        with self.assertRaises(ValueError):module.validate_mechanism_plan(p)

if __name__=='__main__':unittest.main()
