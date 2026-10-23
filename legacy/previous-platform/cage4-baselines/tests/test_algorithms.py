import unittest
try:
 import numpy as np
 from cage4_baselines.core.types import PolicyInput
 from cage4_baselines.algorithms import create
 HAS_NUMPY=True
except ImportError:HAS_NUMPY=False

@unittest.skipUnless(HAS_NUMPY,'numpy required')
class PolicyTests(unittest.TestCase):
 def inputs(self):
  return {f'blue_agent_{i}':PolicyInput(np.ones(64,dtype=np.float32),np.array([False,True,True]),('bad','Sleep','Monitor')) for i in range(5)}
 def config(self):
  return dict(seed=1,device='cpu',lr=.003,gamma=.99,entropy_coef=.01,plugins=[])
 def test_baselines_obey_mask(self):
  inputs=self.inputs()
  for name in ('random','sleep'):
   policy=create(name,inputs,self.config())
   for _ in range(10):
    acts=policy.act(inputs)
    self.assertTrue(all(inputs[a].mask[i] for a,i in acts.items()))
    if name=='sleep':self.assertEqual(set(acts.values()),{1})
 def test_unimplemented_algorithm_rejected(self):
  with self.assertRaises(ValueError):create('mappo',self.inputs(),self.config())
 def test_actor_updates_weights_and_evaluation_does_not(self):
  try:import torch
  except ImportError:self.skipTest('torch optional in lightweight CI')
  inputs=self.inputs();policy=create('actor_critic',inputs,self.config())
  before={a:[p.detach().clone() for p in m.parameters()] for a,m in policy.models.items()}
  for _ in range(3):
   policy.act(inputs,training=True);policy.observe({a:1. for a in inputs})
  losses=policy.update()
  self.assertEqual(len(losses),5)
  for a in inputs:
   self.assertEqual(set(losses[a]),{'actor_loss','critic_loss','entropy','loss'})
   self.assertTrue(any(not torch.equal(x,y) for x,y in zip(before[a],policy.models[a].parameters())))
  weights={a:[p.detach().clone() for p in m.parameters()] for a,m in policy.models.items()}
  for _ in range(3):policy.act(inputs,training=False)
  self.assertTrue(all(not x for x in policy.memory.values()))
  self.assertTrue(all(torch.equal(x,y) for a,m in policy.models.items() for x,y in zip(weights[a],m.parameters())))
if __name__=='__main__':unittest.main()
