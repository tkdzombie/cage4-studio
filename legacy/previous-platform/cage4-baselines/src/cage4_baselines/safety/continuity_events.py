"""Strict online first-hit monitor for the separately versioned OT contract."""
from copy import deepcopy
from .continuity_contract import evaluate
from .service_audit import PHASE_ZONE, summarize


def undiscounted_returns(values, terminals):
    if len(values) != len(terminals) or not values or not terminals[-1]:
        raise ValueError('Expected complete finite episodes')
    result = [0.] * len(values)
    running = 0.
    for t in reversed(range(len(values))):
        if terminals[t]: running = 0.
        running = float(values[t]) + running
        result[t] = running
    return result


class ContinuityEvents:
    def __init__(self, protected, contract):
        self.protected, self.contract = deepcopy(protected), deepcopy(contract)
        self.by_phase = {p: {x['host'] for x in protected if x['zone']==PHASE_ZONE[p]} for p in (1,2)}
        if any(not names for names in self.by_phase.values()) or len({x['host'] for x in protected}) != len(protected):
            raise ValueError('Invalid protected inventory')
        if contract['expected_phase_samples'] != {'1':167,'2':165} or contract['expected_episode_transitions'] != 499:
            raise ValueError('Unexpected phase schedule')
        self.windows = {1:list(range(168,335)),2:list(range(335,500))}
        evaluate(dict(schema='cage4.ot-service-audit/1',profile=contract['inventory_profile'],services=[
            dict(phase=p,host=h,expected_samples=len(self.windows[p]),unknown_samples=0,max_observed_unavailable_run=0)
            for p,hosts in self.by_phase.items() for h in hosts]),contract)
        fraction=contract['max_run_fraction']
        self.limits={p:len(w)*fraction['numerator']//fraction['denominator'] for p,w in self.windows.items()}
        self.step, self.triggered, self.first_hit = 0, False, None
        self.runs, self.rows, self.costs = {}, [], []

    def push(self, step, phase, rows):
        expected_phase = 0 if step <= 167 else (1 if step <=334 else 2)
        if type(step) is not int or step != self.step+1 or step>499 or phase != expected_phase:
            raise ValueError('Skipped, repeated or wrong-phase transition')
        names = self.by_phase.get(phase,set())
        if len(rows)!=len(names) or {r['host'] for r in rows} != names:
            raise ValueError('Missing/duplicate protected snapshot')
        crossed = False
        for row in rows:
            if row['phase']!=phase or row['step']!=step:
                raise ValueError('Snapshot identity mismatch')
            hp,sp,active=row['host_present'],row['service_present'],row['active']
            if type(hp) is not bool or type(sp) is not bool or (sp and not hp) or (sp and type(active) is not bool) or (not sp and active is not None):
                raise ValueError('Unknown or inconsistent service status')
            key=(phase,row['host']);unavailable=not sp or not active
            self.runs[key]=self.runs.get(key,0)+1 if unavailable else 0
            crossed |= self.runs[key]>self.limits[phase]
        cost=int(crossed and not self.triggered)
        if cost:self.triggered=True;self.first_hit=step
        self.rows.extend(deepcopy(rows));self.costs.append(cost);self.step=step
        return cost

    def finalize(self):
        if self.step!=499:raise ValueError('Incomplete episode cannot pass continuity verification')
        measurement=summarize(self.rows,self.protected,self.windows)
        label=evaluate(measurement,self.contract)
        if label['episode_violation'] is not self.triggered or sum(self.costs)!=int(self.triggered):
            raise AssertionError('Online first-hit cost and offline event disagree')
        return dict(measurement=measurement,label=label,first_hit=self.first_hit,event_cost=sum(self.costs))
