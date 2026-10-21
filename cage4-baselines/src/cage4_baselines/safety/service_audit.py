"""Researcher-only OT measurement; no cost, budget or actor integration.

The candidate profile covers every initially deployed OT service in the current
mission zone. This explicit synthetic profile is not a validated business SLA.
"""
import math

PHASE_ZONE = {1: 'operational_zone_a_subnet', 2: 'operational_zone_b_subnet'}


def inventory(base):
    """Call after reset, before actions; freeze this list for the whole episode."""
    from CybORG.Shared.Enums import ProcessName
    state = base.environment_controller.state
    result = []
    for hostname, host in sorted(state.hosts.items()):
        zone = state.hostname_subnet_map[hostname].value
        service = host.services.get(ProcessName.OTSERVICE)
        if zone in PHASE_ZONE.values() and service is not None:
            result.append(dict(host=hostname, zone=zone, service='OTSERVICE',
                               role='server' if '_server_host_' in hostname else 'user',
                               initial_active=bool(service.active),
                               initial_reliability=float(service.get_service_reliability())))
    if set(x['zone'] for x in result) != set(PHASE_ZONE.values()):
        raise ValueError('Expected initially deployed OT services in both mission zones')
    return result


def snapshot(base, protected, step):
    """Live full-state audit only; never pass this result to a Blue actor."""
    from CybORG.Shared.Enums import ProcessName
    state = base.environment_controller.state
    phase = int(state.mission_phase)
    rows = []
    for item in protected:
        if item['zone'] != PHASE_ZONE.get(phase):
            continue
        host = state.hosts.get(item['host'])
        service = host.services.get(ProcessName.OTSERVICE) if host is not None else None
        rows.append(dict(step=step, phase=phase, host=item['host'],
                         host_present=host is not None, service_present=service is not None,
                         active=bool(service.active) if service is not None else None,
                         reliability=float(service.get_service_reliability()) if service is not None else None))
    return rows


def summarize(rows, protected, phase_steps):
    """Compare against explicit expected executed steps; missing rows stay unknown.

    Counts are sampled states, not physical-time downtime or independent trials.
    No aggregate safe/unsafe label or cost scalar is produced.
    """
    items = {x['host']: x for x in protected}
    if not items or len(items) != len(protected):
        raise ValueError('Expected a nonempty inventory with unique hosts')
    for item in protected:
        baseline = item['initial_reliability']
        if item['zone'] not in PHASE_ZONE.values() or type(baseline) not in (float,int) or not math.isfinite(baseline) or not 0 <= baseline <= 100:
            raise ValueError('Invalid inventory zone or initial reliability')
    expected = {}
    for phase, steps in phase_steps.items():
        if type(phase) is not int or phase not in PHASE_ZONE:
            raise ValueError('Only mission phases 1 and 2 are supported')
        steps = list(steps)
        if not steps or any(type(s) is not int or s < 1 for s in steps):
            raise ValueError('Expected positive integer executed step indices')
        if steps != sorted(set(steps)) or any(b != a+1 for a,b in zip(steps,steps[1:])):
            raise ValueError('Expected ordered contiguous phase steps')
        if not any(x['zone'] == PHASE_ZONE[phase] for x in protected):
            raise ValueError('No protected services for phase')
        expected[phase] = steps
    if not expected:
        raise ValueError('Expected at least one mission phase')
    if len({s for ss in expected.values() for s in ss}) != sum(map(len, expected.values())):
        raise ValueError('Phase windows overlap')
    if 1 in expected and 2 in expected and expected[1][-1] >= expected[2][0]:
        raise ValueError('Mission phase windows are out of order')
    indexed = {}
    for row in rows:
        phase, step, host = row['phase'], row['step'], row['host']
        if type(phase) is not int or type(step) is not int:
            raise ValueError('Sample phase and step must be integer indices')
        if phase not in expected or step not in expected[phase] or host not in items:
            raise ValueError('Sample outside declared phase/window/inventory')
        if items[host]['zone'] != PHASE_ZONE[phase]:
            raise ValueError('Sample is not from the protected mission zone')
        key = (phase, step, host)
        if key in indexed:
            raise ValueError('Duplicate service snapshot')
        hp, sp = row['host_present'], row['service_present']
        if type(hp) is not bool or type(sp) is not bool or (sp and not hp):
            raise ValueError('Invalid host/service presence')
        if sp:
            rel = row['reliability']
            if type(row['active']) is not bool or type(rel) not in (float,int) or not math.isfinite(rel) or not 0 <= rel <= 100:
                raise ValueError('Invalid service status or reliability')
        elif row['active'] is not None or row['reliability'] is not None:
            raise ValueError('Absent service must not invent activity/reliability')
        indexed[key] = row
    result = []
    for phase, steps in sorted(expected.items()):
        for host, item in sorted(items.items()):
            if item['zone'] != PHASE_ZONE[phase]:
                continue
            unknown = absent = inactive = degraded = longest = run = 0
            first_unavailable = last_unavailable = None
            for i, step in enumerate(steps):
                row = indexed.get((phase, step, host))
                unavailable = None
                if row is None:
                    unknown += 1
                else:
                    absent += int(not row['service_present'])
                    inactive += int(row['service_present'] and not row['active'])
                    degraded += int(row['service_present'] and row['reliability'] < item['initial_reliability'])
                    unavailable = not row['service_present'] or not row['active']
                if i == 0:
                    first_unavailable = unavailable
                last_unavailable = unavailable
                run = run+1 if unavailable is True else 0
                longest = max(longest, run)
            result.append(dict(phase=phase,host=host,expected_samples=len(steps),
                               observed_samples=len(steps)-unknown,unknown_samples=unknown,
                               absent_service_samples=absent,inactive_service_samples=inactive,
                               degraded_service_samples=degraded,
                               max_observed_unavailable_run=longest,
                               longest_run_is_lower_bound=unknown>0,
                               unavailable_at_window_start=first_unavailable,
                               unavailable_at_window_end=last_unavailable))
    return dict(schema='cage4.ot-service-audit/1',profile='all-initial-ot-in-active-mission-zone/candidate-1',
                unit='post-transition service snapshots; no physical-time mapping',
                complete=all(x['unknown_samples']==0 for x in result),services=result)
