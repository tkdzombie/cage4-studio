import json
from pathlib import Path
import unittest
from cage4_baselines.safety.continuity_events import ContinuityEvents, undiscounted_returns


class ContinuityEventTests(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((Path(__file__).resolve().parents[1]/'configs/safety/ot-continuity-v1.json').read_text())
        self.protected=[dict(host=h,zone='operational_zone_'+z+'_subnet',initial_reliability=100.) for h,z in [('a','a'),('b','b')]]

    def rows(self,step,active):
        phase=0 if step<=167 else (1 if step<=334 else 2)
        return phase,([] if phase==0 else [dict(step=step,phase=phase,host='a' if phase==1 else 'b',host_present=True,service_present=True,active=active,reliability=100.)])

    def rollout(self,unavailable):
        monitor=ContinuityEvents(self.protected,self.contract)
        costs=[]
        for step in range(1,500):
            phase,rows=self.rows(step,not unavailable(step))
            costs.append(monitor.push(step,phase,rows))
        return monitor,costs,monitor.finalize()

    def test_first_hit_once_even_if_both_phases_fail(self):
        m,c,r=self.rollout(lambda s:s>=168)
        self.assertEqual(sum(c),1)
        self.assertEqual(r['first_hit'],184)
        self.assertEqual(c[183],1)
        self.assertTrue(r['label']['phases'][2]['violation'])

    def test_recovery_at_limit_and_independent_mission_counters(self):
        m,c,r=self.rollout(lambda s:168<=s<=183 or 335<=s<=350)
        self.assertEqual(sum(c),0)
        self.assertIs(r['label']['episode_violation'],False)

    def test_prior_damage_counts_from_protected_window_start(self):
        m,c,r=self.rollout(lambda s:s<=200)
        self.assertEqual(r['first_hit'],184)
        self.assertTrue(r['label']['episode_violation'])

    def test_fresh_episode_does_not_inherit_latch(self):
        self.rollout(lambda s:s>167)
        self.assertEqual(self.rollout(lambda s:False)[2]['event_cost'],0)

    def test_missing_or_wrong_order_data_rejected(self):
        m=ContinuityEvents(self.protected,self.contract)
        with self.assertRaises(ValueError):m.push(2,0,[])
        with self.assertRaises(ValueError):m.finalize()
        for s in range(1,168):m.push(s,0,[])
        with self.assertRaises(ValueError):m.push(168,1,[])

    def test_mc_returns_end_at_each_real_terminal(self):
        self.assertEqual(undiscounted_returns([0,1,0,0],[False,True,False,True]),[1,1,0,0])
        self.assertEqual(undiscounted_returns([1,0,0],[False,False,True]),[1,0,0])
        with self.assertRaises(ValueError):undiscounted_returns([1],[False])


if __name__=='__main__':unittest.main()
