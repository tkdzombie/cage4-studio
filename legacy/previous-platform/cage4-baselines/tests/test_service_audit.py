import unittest
from cage4_baselines.safety.service_audit import summarize


class ServiceAuditTests(unittest.TestCase):
    def setUp(self):
        self.a = dict(host='a',zone='operational_zone_a_subnet',initial_reliability=100.)
        self.b = dict(host='b',zone='operational_zone_b_subnet',initial_reliability=100.)

    def row(self, step, active=True, reliability=100., host='a', phase=1):
        return dict(step=step,phase=phase,host=host,host_present=True,
                    service_present=True,active=active,reliability=reliability)

    def test_unknown_breaks_run_without_becoming_healthy(self):
        d=summarize([self.row(1,False),self.row(3,False)], [self.a], {1:[1,2,3]})
        self.assertFalse(d['complete'])
        s=d['services'][0]
        self.assertEqual(s['unknown_samples'],1)
        self.assertEqual(s['max_observed_unavailable_run'],1)
        self.assertTrue(s['longest_run_is_lower_bound'])
        self.assertTrue(s['unavailable_at_window_end'])

    def test_missing_protected_service_is_not_dropped_from_denominator(self):
        r=self.row(1);r.update(host_present=False,service_present=False,active=None,reliability=None)
        s=summarize([r], [self.a], {1:[1]})['services'][0]
        self.assertEqual(s['absent_service_samples'],1)
        self.assertEqual(s['inactive_service_samples'],0)
        self.assertEqual(s['max_observed_unavailable_run'],1)

    def test_phase_and_host_results_do_not_offset_each_other(self):
        rows=[self.row(1,False),self.row(2,False),self.row(3,host='b',phase=2)]
        s=summarize(rows,[self.a,self.b],{1:[1,2],2:[3]})['services']
        self.assertEqual([x['max_observed_unavailable_run'] for x in s],[2,0])
        self.assertTrue(s[0]['unavailable_at_window_end'])
        self.assertFalse(s[1]['unavailable_at_window_start'])

    def test_degradation_and_inactivity_are_distinct(self):
        s=summarize([self.row(1,True,80),self.row(2,False,100)], [self.a], {1:[1,2]})['services'][0]
        self.assertEqual(s['degraded_service_samples'],1)
        self.assertEqual(s['inactive_service_samples'],1)

    def test_no_observations_does_not_pass(self):
        d=summarize([], [self.a], {1:[1,2]})
        self.assertFalse(d['complete'])
        self.assertIsNone(d['services'][0]['unavailable_at_window_end'])

    def test_invalid_or_duplicate_evidence_is_rejected(self):
        for rows,windows in [([self.row(1),self.row(1)],{1:[1]}),
                             ([self.row(1,reliability=float('nan'))],{1:[1]}),
                             ([self.row(1,host='b')],{1:[1]}),
                             ([],{1:[1,3]}),([],{}),([],{1:[1],2:[1]})]:
            with self.subTest(rows=rows,windows=windows):
                with self.assertRaises(ValueError):summarize(rows,[self.a,self.b],windows)


if __name__=='__main__':
    unittest.main()
