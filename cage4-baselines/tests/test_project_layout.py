import ast
import importlib
from pathlib import Path
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]

class ProjectContractTests(unittest.TestCase):
 def test_all_documented_config_profiles_validate(self):
  from cage4_baselines.config import load
  profiles={'train/actor_critic.yaml':'actor_critic','evaluate/actor_critic.yaml':'actor_critic',
            'baselines/random.yaml':'random','baselines/sleep.yaml':'sleep'}
  for profile,algorithm in profiles.items():
   with self.subTest(profile=profile):
    self.assertEqual(load(ROOT/'configs'/profile)['algorithm'],algorithm)
  self.assertEqual(load(ROOT/'configs/evaluate/actor_critic.yaml')['episodes'],5)
 def test_config_typo_is_rejected(self):
  from cage4_baselines.config import load
  with tempfile.TemporaryDirectory() as folder:
   p=Path(folder)/'bad.yaml';p.write_text('learning_rate_typo: 0.01\n')
   with self.assertRaises(ValueError):load(p)
 def test_legacy_imports_share_the_new_implementations(self):
  pairs=[('adapter','envs.adapter','Cage4Adapter'),('adapter','core.types','PolicyInput'),
         ('runner','experiments.runner','run'),('engine','envs.factory','create_env'),
         ('observations','envs.observations','encode'),('telemetry','recording.telemetry','frame')]
  for old,new,symbol in pairs:
   self.assertIs(getattr(importlib.import_module('cage4_baselines.'+old),symbol),
                 getattr(importlib.import_module('cage4_baselines.'+new),symbol))
 def test_algorithms_do_not_import_simulation_or_telemetry(self):
  package=ROOT/'src/cage4_baselines/algorithms'
  for path in package.glob('*.py'):
   tree=ast.parse(path.read_text())
   imports=[]
   for node in ast.walk(tree):
    if isinstance(node,ast.ImportFrom):imports.append(node.module or '')
    elif isinstance(node,ast.Import):imports.extend(a.name for a in node.names)
   for module in imports:
    self.assertFalse(set(module.split('.')) & {'envs','recording','telemetry','CybORG'},(path.name,module))
if __name__=='__main__':unittest.main()
