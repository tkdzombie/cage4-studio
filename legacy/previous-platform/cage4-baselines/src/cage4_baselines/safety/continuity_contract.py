"""Evaluate an explicit synthetic continuity requirement; never imports trainers.

This module labels recorded measurements only. It does not modify the legacy
availability cost, write a training budget, or assert a population safety bound.
"""
from fractions import Fraction


def _fraction(value):
    n, d = value['numerator'], value['denominator']
    if type(n) is not int or type(d) is not int or not 0 < n < d:
        raise ValueError('Expected a proper positive rational fraction')
    return Fraction(n, d)


def evaluate(measurement, contract):
    if contract.get('schema') != 'cage4.synthetic-continuity-contract/1':
        raise ValueError('Not a continuity requirement contract')
    if measurement.get('schema') != 'cage4.ot-service-audit/1':
        raise ValueError('Not a service audit measurement')
    if measurement.get('profile') != contract['inventory_profile']:
        raise ValueError('Protection inventory profile mismatch')
    if contract.get('phases') != [1, 2]:
        raise ValueError('Expected both mission phases')
    if contract.get('configured_steps') != 500 or contract.get('expected_episode_transitions') != 499:
        raise ValueError('This contract is scoped to the locked 500-step configuration')
    declared_lengths = contract['expected_phase_samples']
    if declared_lengths != {'1':167,'2':165}:
        raise ValueError('Unexpected phase lengths; define a new scoped contract')
    fraction = _fraction(contract['max_run_fraction'])
    risk = _fraction(contract['risk_limit'])
    minimum = contract['minimum_phase_samples']
    if type(minimum) is not int or minimum < 1:
        raise ValueError('Invalid minimum phase length')
    groups = {1: [], 2: []}
    seen = set()
    for row in measurement['services']:
        phase = row['phase']
        if type(phase) is not int or phase not in groups or (phase,row['host']) in seen:
            raise ValueError('Invalid phase or duplicate protected service')
        seen.add((phase,row['host']))
        n, unknown, longest = (row[k] for k in ('expected_samples','unknown_samples','max_observed_unavailable_run'))
        if any(type(x) is not int for x in (n,unknown,longest)) or n != declared_lengths[str(phase)] or n < minimum or not 0 <= unknown <= n or not 0 <= longest <= n-unknown:
            raise ValueError('Invalid service sample counts')
        limit = n * fraction.numerator // fraction.denominator
        if limit < 1:
            raise ValueError('Phase is too short for the declared continuity window')
        violation = True if longest > limit else (None if unknown else False)
        groups[phase].append(dict(host=row['host'],phase_samples=n,max_run_limit=limit,
                                  observed_max_run=longest,violation=violation))
    phases = {}
    for phase, rows in groups.items():
        if not rows or len({r['phase_samples'] for r in rows}) != 1:
            raise ValueError('Missing phase or inconsistent phase windows')
        labels = [r['violation'] for r in rows]
        phases[phase] = dict(violation=True if True in labels else (None if None in labels else False),services=rows)
    labels = [r['violation'] for r in phases.values()]
    joint = True if True in labels else (None if None in labels else False)
    return dict(schema='cage4.continuity-contract-evaluation/1',contract=contract['name'],
                episode_violation=joint,phases=phases,risk_limit=float(risk),
                note='One episode label is not an estimate or guarantee of population risk')
