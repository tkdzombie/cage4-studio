import unittest
import numpy as np
from cage4_baselines.envs.observations import StructuredObservation, HOST_FIELDS

class StructuredTests(unittest.TestCase):
 def encoder(self):
  return StructuredObservation(['a_user_host_0','a_server_host_0'], ['Analyse','Remove','Sleep'], 3, 500, 0)
 def rows(self, e, obs, step):
  return e.encode(obs,step)[-3*len(e.fields):].reshape(3,-1)
 def test_initial_inventory_not_an_alert_and_identity_numbers_invariant(self):
  a={'a_user_host_0':{'Processes':[{'PID':123}], 'Interface':[{'ip_address':'10.0.0.1'}]}}
  b={'a_user_host_0':{'Processes':[{'PID':456}], 'Interface':[{'ip_address':'10.99.2.8'}]}}
  self.assertTrue(np.array_equal(self.encoder().encode(a,0),self.encoder().encode(b,0)))
  r=self.rows(self.encoder(),a,0)
  self.assertEqual(r[0,0],1);self.assertEqual(r[0,3],0)
  self.assertFalse(r[2].any())
 def test_history_recovery_and_episode_reset(self):
  e=self.encoder()
  r=self.rows(e,{'a_user_host_0':{'Processes':[{'PID':123}]}},1)
  self.assertEqual(r[0,5],1)
  r=self.rows(e,{'success':'FALSE'},2)
  self.assertEqual(r[0,3],0);self.assertEqual(r[0,5],1)
  r=self.rows(e,{'success':'TRUE','action':'Remove a_user_host_0'},3)
  self.assertEqual(r[0,5],1)  # Remove cannot certify privileged process removal.
  r=self.rows(e,{'success':'TRUE','action':'Restore a_user_host_0'},4)
  self.assertEqual(r[0,5],0)
  e.reset();self.assertEqual(self.rows(e,{},0)[0,0],0)
 def test_order_is_fixed_and_foreign_truth_ignored(self):
  obs={'a_server_host_0':{'Processes':[{'Connections':[{}]}]},'red_truth':{'infected':True}}
  first=self.encoder().encode(obs,1)
  obs['red_truth']['infected']=False
  self.assertTrue(np.array_equal(first,self.encoder().encode(obs,1)))
  self.assertEqual(len(first),self.encoder().schema()['dimension'])
 def test_successful_analysis_only_and_finite_messages(self):
  e=self.encoder()
  r=self.rows(e,{'success':'FALSE','action':'Analyse a_user_host_0','a_user_host_0':{'Files':[{}]}},1)
  self.assertEqual(r[0,7],0)
  r=self.rows(e,{'success':'TRUE','action':'Analyse a_user_host_0','a_user_host_0':{'Files':[{}]}},2)
  self.assertEqual(r[0,7],1);self.assertAlmostEqual(r[0,8],.1)
  with self.assertRaises(ValueError):e.encode({'message':[[float('nan')]*8]*4},3)

class BeliefTests(unittest.TestCase):
 def encoder(self, **options):
  return StructuredObservation(['a_user_host_0'], ['Analyse','Remove','Restore','Sleep'], 1, 500, 0, options)
 def test_age_confidence_missing_reobserve_and_channel_masks(self):
  e=self.encoder();F=e.fields
  def row(o,t):return e.encode(o,t)[-len(F):]
  r=row({'a_user_host_0':{'Processes':[{'PID':9}]}},1)
  self.assertEqual(r[F.index('observed/process')],1)
  self.assertEqual(r[F.index('age_linear/process')],0)
  self.assertEqual(r[F.index('confidence/process')],1)
  r=row({},2)
  self.assertEqual(r[F.index('observed/process')],0)
  self.assertEqual(r[F.index('process_history')],1)
  self.assertAlmostEqual(r[F.index('age_linear/process')],1/500)
  confidence=r[F.index('confidence/process')]
  r=row({},3);self.assertLess(r[F.index('confidence/process')],confidence)
  r=row({'a_user_host_0':{'Processes':[{'PID':10}]}},4)
  self.assertEqual(r[F.index('age_linear/process')],0)
  self.assertEqual(r[F.index('confidence/process')],1)
 def test_dropout_does_not_mutate_observation_or_invent_analysis_negative(self):
  from copy import deepcopy
  obs={'success':'TRUE','action':'Analyse a_user_host_0','a_user_host_0':{'Files':[{}]}}
  before=deepcopy(obs);e=self.encoder(dropout=1.)
  x=e.encode(obs,1);self.assertEqual(obs,before)
  self.assertEqual(x[-len(e.fields)+e.fields.index('analysed')],0)
 def test_delay_uses_source_time_and_seeded_noise_is_reproducible(self):
  a=self.encoder(delay=1);F=a.fields
  obs={'a_user_host_0':{'Processes':[{'PID':1}]}}
  a.encode(obs,1);x=a.encode({},2)[-len(F):]
  self.assertAlmostEqual(x[F.index('age_linear/process')],1/500)
  self.assertLess(x[F.index('confidence/process')],1)
  a=self.encoder(seed=5,dropout=.2,noise=.3);b=self.encoder(seed=5,dropout=.2,noise=.3)
  for t in range(20):self.assertTrue(np.array_equal(a.encode(obs,t),b.encode(obs,t)))
 def test_graph_indices_and_ablation_fields_are_explicit(self):
  e=self.encoder();e.encode({},0);g=e.graph()
  self.assertEqual(g['edge_index'].shape,(2,2))
  self.assertEqual(g['metadata']['host_indices']['a_user_host_0'],0)
  self.assertEqual(g['metadata']['subnet_indices']['a'],1)
  self.assertEqual(g['node_features'].shape,(4,len(e.fields)))
  e=self.encoder(mode='persistent');e.encode({'a_user_host_0':{'Processes':[{}]}},1)
  x=e.encode({},2)[-len(e.fields):]
  self.assertEqual(x[e.fields.index('process_history')],1)
  for key in e.fields:
   if key.startswith(('age_', 'confidence/')):self.assertEqual(x[e.fields.index(key)],0)
