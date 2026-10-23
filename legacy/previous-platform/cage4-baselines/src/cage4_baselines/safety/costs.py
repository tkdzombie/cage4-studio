"""CAGE4 availability cost v1, aligned with the installed BlueRewardMachine.

This is a proposed CMDP extension, not an official CAGE4 constraint. Only the
scalar cost is supplied to constrained learners; event details remain telemetry.
"""
from dataclasses import dataclass, asdict
from math import isfinite

COST_VERSION = 'cage4.availability/1'


@dataclass(frozen=True)
class Workload:
    weight: float
    failed: bool
    kind: str
    zone: str


def aggregate(workloads):
    attempted = failed = 0.
    counts = {'eligible_tasks': 0, 'failed_tasks': 0, 'zero_weight_tasks': 0}
    for task in workloads:
        if not isfinite(task.weight) or task.weight < 0:
            raise ValueError('Task weights must be finite and nonnegative')
        if task.weight == 0:
            counts['zero_weight_tasks'] += 1
            continue
        attempted += task.weight
        counts['eligible_tasks'] += 1
        if task.failed:
            failed += task.weight
            counts['failed_tasks'] += 1
    return dict(version=COST_VERSION, cost=failed / attempted if attempted else 0.,
                attempted_weight=attempted, failed_weight=failed,
                no_eligible_tasks=not bool(attempted), **counts)


def measure(base):
    """Call immediately after step, before the next action mutates controller state.

    Uses the same active-session filter, first executed action, first observation,
    source subnet and current mission phase as the official reward calculator.
    Unknown/absent outcomes fail loudly instead of being counted as successes.
    """
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
    from CybORG.Simulator.Actions import GreenLocalWork, GreenAccessService
    controller = base.environment_controller
    state = controller.state
    weights = BlueRewardMachine('Blue').get_phase_rewards(state.mission_phase)
    tasks = []
    inactive = 0
    for name, actions in controller.action.items():
        if 'green' not in name or not actions:
            continue
        action = actions[0]
        if not isinstance(action, (GreenLocalWork, GreenAccessService)):
            continue
        if not any(s.active for s in state.sessions[name].values()):
            inactive += 1
            continue
        host = state.ip_addresses[action.ip_address]
        zone = state.hostname_subnet_map[host].value
        success = controller.observation[name].observations[0].data['success']
        # CybORG TrinaryEnum implements equality to True/False.
        if success != True and success != False:
            raise RuntimeError(f'Unresolved Green outcome: {success!r}')
        kind = 'LWF' if isinstance(action, GreenLocalWork) else 'ASF'
        tasks.append(Workload(abs(weights[zone][kind]), success == False, kind, zone))
    result = aggregate(tasks)
    result.update(phase=int(state.mission_phase), inactive_tasks=inactive)
    return result


def summarize(rows):
    if not rows:
        raise ValueError('Cannot summarize an empty episode')
    n = len(rows)
    failed = sum(x['failed_weight'] for x in rows)
    attempted = sum(x['attempted_weight'] for x in rows)
    total = sum(x['cost'] for x in rows)
    return dict(version=COST_VERSION, episode_cost=total, mean_cost=total/n,
                request_weighted_failure_rate=failed/attempted if attempted else None,
                attempted_weight=attempted, failed_weight=failed,
                no_task_steps=sum(x['no_eligible_tasks'] for x in rows),
                inactive_tasks=sum(x.get('inactive_tasks', 0) for x in rows), steps=n)
