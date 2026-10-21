import copy
import json
from pathlib import Path
import unittest
from cage4_baselines.safety.continuity_contract import evaluate


class ContinuityContractTests(unittest.TestCase):
    def setUp(self):
        self.contract=json.loads((Path(__file__).resolve().parents[1]/'configs/safety/ot-continuity-v1.json').read_text())

    def measurement(self,a=16,b=16,unknown=0):
        return dict(schema='cage4.ot-service-audit/1',profile=self.contract['inventory_profile'],services=[
            dict(phase=1,host='a',expected_samples=167,unknown_samples=unknown,max_observed_unavailable_run=a),
            dict(phase=2,host='b',expected_samples=165,unknown_samples=0,max_observed_unavailable_run=b)])

    def test_exact_boundary_and_floor_are_explicit(self):
        result=evaluate(self.measurement(),self.contract)
        self.assertIs(result['episode_violation'],False)
        self.assertEqual(result['phases'][1]['services'][0]['max_run_limit'],16)
        self.assertEqual(result['phases'][2]['services'][0]['max_run_limit'],16)
        self.assertIs(evaluate(self.measurement(a=17,b=0),self.contract)['episode_violation'],True)

    def test_other_phase_success_does_not_cancel_violation(self):
        result=evaluate(self.measurement(a=167,b=0),self.contract)
        self.assertIs(result['phases'][2]['violation'],False)
        self.assertIs(result['episode_violation'],True)

    def test_missing_evidence_is_unknown_unless_violation_confirmed(self):
        self.assertIsNone(evaluate(self.measurement(a=0,b=0,unknown=1),self.contract)['episode_violation'])
        self.assertIs(evaluate(self.measurement(a=17,b=0,unknown=1),self.contract)['episode_violation'],True)

    def test_any_service_can_trigger_without_averaging(self):
        m=self.measurement(a=0,b=0)
        m['services'].append(dict(phase=1,host='critical',expected_samples=167,unknown_samples=0,max_observed_unavailable_run=100))
        self.assertIs(evaluate(m,self.contract)['episode_violation'],True)

    def test_missing_phase_and_mismatched_inventory_rejected(self):
        m=self.measurement();m['services'].pop()
        with self.assertRaises(ValueError):evaluate(m,self.contract)
        m=self.measurement();m['profile']='other'
        with self.assertRaises(ValueError):evaluate(m,self.contract)

    def test_invalid_contract_or_counts_cannot_silently_pass(self):
        for field,value in [('risk_limit',{'numerator':0,'denominator':20}),('max_run_fraction',{'numerator':1,'denominator':0}),('phases',[1])]:
            c=copy.deepcopy(self.contract);c[field]=value
            with self.assertRaises(ValueError):evaluate(self.measurement(),c)
        m=self.measurement();m['services'][0]['max_observed_unavailable_run']=168
        with self.assertRaises(ValueError):evaluate(m,self.contract)
        m=self.measurement(a=0,b=0);m['services'][0]['expected_samples']=100
        with self.assertRaises(ValueError):evaluate(m,self.contract)


if __name__=='__main__':unittest.main()
