import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('evaluate_baseline',Path(__file__).resolve().parents[1]/'ops/evaluate_baseline.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class EvaluationConfigTests(unittest.TestCase):
 def test_recorded_scientific_notation_remains_numeric(self):
  original=dict(algorithm='mappo',seed=101,steps=500,episodes=2000,
                hyperparameters=dict(lr=5e-5,critic_lr=5e-5),deterministic_eval=True)
  with tempfile.TemporaryDirectory() as t:
   (Path(t)/'config.json').write_text(json.dumps(original))
   recovered=module.read_training_config(t)
   self.assertEqual(recovered,original)
   self.assertIsInstance(recovered['hyperparameters']['lr'],float)
