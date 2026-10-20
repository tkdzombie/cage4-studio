"""Real spawn/learner integration checks, skipped in lightweight CI."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

REAL=bool(importlib.util.find_spec('CybORG') and importlib.util.find_spec('torch'))

@unittest.skipUnless(REAL,'real CybORG/Torch runtime required')
class ParallelTests(unittest.TestCase):
 def config(self):
  return dict(algorithm='mappo',seed=101,episodes=4,steps=20,rollout_episodes=4,num_envs=2,torch_threads=1,
              reward_scale=.01,checkpoint_every=3,record_replay=True,snapshot_every=10,deterministic_eval=True,
              hyperparameters=dict(hidden_size=32,layer_N=1,ppo_epoch=2))
 def test_frozen_weights_rng_and_single_vs_spawn_trajectories(self):
  import torch,random
  from cage4_baselines.envs.adapter import Cage4Adapter
  from cage4_baselines.paper.backend import AuthorBaseline,fingerprint
  from cage4_baselines.paper.rollouts import EpisodeCollector,RolloutPool
  c=self.config();torch.set_num_threads(1);torch.manual_seed(101)
  env=Cage4Adapter(seed=101,steps=20,include_graph=False)
  policy=AuthorBaseline(env.reset(),c,Path.home()/'cage4-upstreams',.2474800289213457)
  collector=EpisodeCollector(c,.2474800289213457,19,policy,env)
  jobs=[(1,10200000),(2,10200001)]
  weights=policy.rollout_state();before={a:fingerprint(p.actor) for a,p in policy.policies.items()}
  rng=torch.get_rng_state().clone();rng_py=random.getstate()
  serial=[collector.collect(ep,seed) for ep,seed in jobs]
  self.assertTrue(torch.equal(rng,torch.get_rng_state()));self.assertEqual(rng_py,random.getstate())
  self.assertEqual(before,{a:fingerprint(p.actor) for a,p in policy.policies.items()})
  with tempfile.TemporaryDirectory() as folder:
   with RolloutPool(2,c,Path.home()/'cage4-upstreams',.2474800289213457,19) as pool:
    children=list(pool.processes)
    parallel=pool.collect(weights,jobs,folder)
   self.assertTrue(all(not p.is_alive() for p in children))
   self.assertEqual(len(list(Path(folder).glob('frames/episode_*/step_*.json'))),40)
  for a,b in zip(serial,parallel):
   self.assertEqual(a['row']['official_team_return'],b['row']['official_team_return'])
   self.assertEqual(a['row']['safety'],b['row']['safety'])
   for old,new in zip(a['history'],b['history']):
    self.assertEqual(old[1:],new[1:])
    for agent,record in old[0].items():
     for key,value in record.items():
      if value is None:self.assertIsNone(new[0][agent][key])
      else:np.testing.assert_array_equal(value,new[0][agent][key])
 def test_checkpoint_boundary_retains_full_gradient_batch(self):
  import torch
  from cage4_baselines.paper.__main__ import run
  root=Path(__file__).resolve().parents[1]
  c=self.config();c['record_replay']=False
  with tempfile.TemporaryDirectory() as t:
   folder=run(c,t,Path.home()/'cage4-upstreams',root/'configs/safety/availability-v1.json',smoke=True)
   rows=[json.loads(p.read_text()) for p in sorted((folder/'metrics').rglob('*.json'))]
   self.assertEqual([r['episode'] for r in rows if r['optimizer_update']],[4])
   cp3=torch.load(folder/'checkpoints/episode_000003.pt',weights_only=True)
   cp4=torch.load(folder/'checkpoints/episode_000004.pt',weights_only=True)
   self.assertEqual(cp3['last_optimizer_episode'],0);self.assertEqual(cp3['training_episode_seeds'],[])
   self.assertEqual(cp4['last_optimizer_episode'],4);self.assertEqual(len(cp4['training_episode_seeds']),4)
   self.assertEqual(cp4['total_transitions'],76)
   self.assertTrue(all(v['actor_changed'] and v['critic_changed'] for v in rows[-1]['losses'].values()))
   # More validation episodes than worker slots exercises the second collection
   # chunk; eval must neither accumulate training batches nor change actor weights.
   evaluation=run(c,Path(t)/'evaluation',Path.home()/'cage4-upstreams',root/'configs/safety/availability-v1.json',
                  mode='evaluate',checkpoint=folder/'checkpoints/episode_000004.pt',seeds=[54001,54002,54003],smoke=True)
   self.assertEqual(json.loads((evaluation/'evaluation.json').read_text())['n_seeds'],3)
 def test_worker_failure_is_reported_and_workers_close(self):
  from cage4_baselines.envs.adapter import Cage4Adapter
  from cage4_baselines.paper.backend import AuthorBaseline
  from cage4_baselines.paper.rollouts import RolloutPool
  c=self.config();env=Cage4Adapter(seed=101,steps=20,include_graph=False)
  p=AuthorBaseline(env.reset(),c,Path.home()/'cage4-upstreams',.2474800289213457)
  with RolloutPool(1,c,Path.home()/'cage4-upstreams',.2474800289213457,19) as pool:
   children=list(pool.processes)
   with self.assertRaises(RuntimeError):pool.collect(p.rollout_state(),[(1,-1)],None)
  self.assertTrue(all(not p.is_alive() for p in children))
 def test_graph_export_does_not_change_flat_input(self):
  from cage4_baselines.envs.adapter import Cage4Adapter
  a=Cage4Adapter(seed=42,steps=20);b=Cage4Adapter(seed=42,steps=20,include_graph=False)
  x=a.reset();y=b.reset()
  for name in x:
   np.testing.assert_array_equal(x[name].observation,y[name].observation)
   np.testing.assert_array_equal(x[name].mask,y[name].mask)
   self.assertIsNone(y[name].graph)
