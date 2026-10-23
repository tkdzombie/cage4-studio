"""Bounded task diagnostics, not a learning baseline or safety certification.

The controller accepts only public policy inputs; hidden state is measured after
actions. All three rules and four development scenarios are fixed before runs.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import time


SEEDS = (54001, 54002, 54003, 54004)
RULES = ('alert_restore', 'mission_barriers', 'all_barriers')


class PublicRule:
    def __init__(self, rule, schemas):
        if rule not in RULES:
            raise ValueError(rule)
        self.rule, self.schemas = rule, schemas
        self.attempted = {a: set() for a in schemas}

    def act(self, inputs):
        result = {}
        for agent, value in inputs.items():
            legal = [(i, s) for i, s in enumerate(value.labels) if value.mask[i]]
            chosen = None
            barrier = self.rule == 'all_barriers' or (self.rule == 'mission_barriers' and agent in ('blue_agent_1', 'blue_agent_3'))
            if barrier:
                chosen = next(((i, s) for i, s in legal if s.startswith('BlockTrafficZone ') and s not in self.attempted[agent]), None)
            if chosen is None:
                schema = self.schemas[agent]
                width = len(schema['hosts'][0]['fields'])
                rows = value.observation[len(schema['head']):].reshape(schema['max_hosts'], width)
                for host in schema['hosts']:
                    fields = dict(zip(host['fields'], rows[host['slot']]))
                    if fields['process_history'] or fields['connection_history']:
                        target = 'Restore ' + host['hostname']
                        chosen = next(((i, s) for i, s in legal if s == target), None)
                        if chosen:
                            break
            if chosen is None:
                chosen = next((i, s) for i, s in legal if s == 'Sleep')
            result[agent] = chosen[0]
            self.attempted[agent].add(chosen[1])
        return result


def fixtures(env):
    """Paired identical resets; no transition or actor update."""
    from CybORG.Shared.Enums import ProcessName
    from CybORG.Simulator.Actions.AbstractActions.Monitor import Monitor
    from CybORG.Simulator.Actions.AbstractActions.Restore import Restore
    healthy = env.base.environment_controller.state
    for host in healthy.hosts.values():
        host.events.network_connections.clear()
        host.events.process_creation.clear()
    a = Monitor(0, 'blue_agent_1').execute(healthy).data
    env.reset(SEEDS[0])
    stopped = env.base.environment_controller.state
    for host in stopped.hosts.values():
        host.events.network_connections.clear()
        host.events.process_creation.clear()
    target = next(h for h in sorted(stopped.hosts) if h.startswith('operational_zone_a') and ProcessName.OTSERVICE in stopped.hosts[h].services)
    stopped.stop_service(target, ProcessName.OTSERVICE)
    b = Monitor(0, 'blue_agent_1').execute(stopped).data
    before = bool(stopped.hosts[target].services[ProcessName.OTSERVICE].active)
    result = Restore(0, 'blue_agent_1', target).execute(stopped)
    return dict(target=target, healthy_and_stopped_monitor_equal=a == b,
                stopped_service_active_before_restore=before,
                direct_restore_success=result.success == True,
                service_active_after_restore=bool(stopped.hosts[target].services[ProcessName.OTSERVICE].active),
                scope='Paired deterministic resets with cleared events; direct execute bypasses scheduling duration; no frequency or timing claim')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--fixtures-only', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.service_audit import inventory, snapshot
    from cage4_baselines.safety.continuity_events import ContinuityEvents
    from cage4_baselines.safety.costs import measure
    contract = json.loads(args.contract.read_text())
    manifest = dict(schema='cage4.ot-task-validity-probe/1', seeds=SEEDS, rules=RULES,
                    max_transitions=0 if args.fixtures_only else 12 * 499,
                    training_updates=0, policy_inputs='legal public PolicyInput only',
                    contract_sha256=hashlib.sha256(args.contract.read_bytes()).hexdigest(),
                    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    runtime=runtime_identity(), started_at=time.time())
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2))
    env = Cage4Adapter(SEEDS[0], 500, include_graph=False)
    env.reset()
    (args.output/'fixtures.json').write_text(json.dumps(fixtures(env), indent=2))
    if args.fixtures_only:
        return
    results = []
    started = time.monotonic()
    for rule in RULES:
        for seed in SEEDS:
            if (args.output/'STOP').exists() or shutil.disk_usage(args.output).free < 30 * 1024**3 or time.monotonic()-started > 3600:
                raise InterruptedError('Diagnostic resource/STOP boundary reached')
            inputs = env.reset(seed)
            controller = PublicRule(rule, env.observation_schema())
            monitor = ContinuityEvents(inventory(env.base), contract)
            total = 0.
            costs, actions = [], Counter()
            for step in range(1, 500):
                if (args.output/'STOP').exists():
                    raise InterruptedError('STOP requested')
                selected = controller.act(inputs)
                actions.update(inputs[a].labels[i].split()[0] for a, i in selected.items())
                inputs, reward, done, truncated = env.step(selected)
                phase = int(env.base.environment_controller.state.mission_phase)
                monitor.push(step, phase, snapshot(env.base, monitor.protected, step))
                total += sum(reward.values()) / len(reward)
                costs.append(measure(env.base)['cost'])
                if all(done.get(a, False) or truncated.get(a, False) for a in selected):
                    break
            result = monitor.finalize()
            results.append(dict(rule=rule, seed=seed, transitions=step, return_=total,
                                event=result['event_cost'], first_hit=result['first_hit'],
                                legacy_cost=sum(costs)/len(costs), actions=dict(actions), ot=result['measurement']))
            temporary = args.output/'results.tmp'
            temporary.write_text(json.dumps(results))
            temporary.replace(args.output/'results.json')
            print(json.dumps({k:v for k,v in results[-1].items() if k != 'ot'}), flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(phase_complete=True, episodes=len(results), transitions=sum(r['transitions'] for r in results))))


if __name__ == '__main__':
    main()
