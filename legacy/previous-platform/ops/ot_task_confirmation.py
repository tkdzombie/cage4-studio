"""Frozen single-rule risk confirmation and fixed task ablations; zero training."""
import argparse
import importlib.util
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace


def upper_bound(events, episodes, alpha=.05):
    """One-sided exact binomial upper endpoint, conditional on iid episodes."""
    if not 0 <= events <= episodes or episodes < 1 or not 0 < alpha < 1:
        raise ValueError('Invalid binomial evidence')
    if events == episodes:
        return 1.
    lo, hi = 0., 1.
    for _ in range(80):
        p = (lo+hi)/2
        cdf = sum(math.comb(episodes,k)*p**k*(1-p)**(episodes-k) for k in range(events+1))
        if cdf > alpha:
            lo = p
        else:
            hi = p
    return (lo+hi)/2


def write(path, obj):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-probe', type=Path, required=True)
    p.add_argument('--contract', type=Path, required=True)
    p.add_argument('--expected-runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('confirmation','ablations','cadence'), required=True)
    args = p.parse_args()
    args.output.mkdir(exist_ok=False)
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.continuity_events import ContinuityEvents
    from cage4_baselines.safety.service_audit import inventory,snapshot
    from cage4_baselines.safety.costs import measure
    from cage4_protocol.store import process_identity
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
    spec = importlib.util.spec_from_file_location('frozen_probe', args.base_probe)
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    runtime = runtime_identity()
    if runtime != json.loads(args.expected_runtime.read_text())['runtime']:
        raise RuntimeError('Source/runtime drift')
    contract = json.loads(args.contract.read_text())
    contract_sha = hashlib.sha256(args.contract.read_bytes()).hexdigest()
    if contract_sha != '542d70e17de17625f821ae37ace340ff4f63697c9c9113bb48ac44d144b34248':
        raise RuntimeError('Contract drift')
    seeds = list(range(55001,55061)) if args.mode == 'confirmation' else list(range(54001,54005))
    rules = (['alert_restore'] if args.mode == 'confirmation' else
             (['restore_interval_8','restore_interval_12','restore_interval_16'] if args.mode=='cadence' else
              ['process_only','operations_only','support_only','restore_interval_20']))
    plan = [(r,s) for r in rules for s in seeds]
    identity = process_identity(os.getpid())
    manifest = dict(schema='cage4.ot-task-confirmation/1',mode=args.mode,plan=plan,
                    single_fixed_policy=args.mode=='confirmation',alpha=.05,
                    selection='alert_restore selected using only prior four development cases; no adjustment within confirmation',
                    risk_inference='Conditional on independent reset scenarios from the specified simulator distribution; not cross-training-seed assurance',
                    runtime=runtime,contract_sha256=contract_sha,
                    base_sha256=hashlib.sha256(args.base_probe.read_bytes()).hexdigest(),
                    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    max_transitions=len(plan)*499,training_updates=0,pid=os.getpid(),identity=identity)
    write(args.output/'manifest.json',manifest)
    shutil.copyfile(args.base_probe,args.output/'frozen-rule.py')
    stopped = False
    def stop(*_):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGINT,stop)
    signal.signal(signal.SIGTERM,stop)
    started = time.monotonic()
    def guard():
        if stopped or (args.output/'STOP').exists() or time.monotonic()-started>3600 or shutil.disk_usage(args.output).free<30*1024**3:
            raise InterruptedError('STOP/resource/time boundary reached')
    rows = []
    write(args.output/'state.json',dict(state='running',pid=os.getpid(),identity=identity,completed=0,total=len(plan)))
    try:
        env = Cage4Adapter(seeds[0],500,include_graph=False)
        for rule,seed in plan:
            guard()
            inputs = env.reset(seed)
            schemas = env.observation_schema()
            controller = probe.PublicRule('alert_restore',schemas)
            monitor = ContinuityEvents(inventory(env.base),contract)
            last_restore = {a:-10000 for a in inputs}
            total = 0.
            costs = []
            decomposition = dict(work_failure=0.,impact=0.,intervention=0.)
            calculator = BlueRewardMachine('Blue')
            restores = {a:0 for a in inputs}
            for step in range(1,500):
                guard()
                public = inputs
                if rule == 'process_only':
                    public = {}
                    for a,value in inputs.items():
                        obs = value.observation.copy()
                        schema = schemas[a]
                        width = len(schema['hosts'][0]['fields'])
                        host_rows = obs[len(schema['head']):].reshape(schema['max_hosts'],width)
                        host_rows[:,schema['hosts'][0]['fields'].index('connection_history')] = 0
                        public[a] = SimpleNamespace(observation=obs,mask=value.mask,labels=value.labels)
                actions = controller.act(public)
                for a,i in list(actions.items()):
                    disable = (rule=='operations_only' and a not in ('blue_agent_1','blue_agent_3')) or (rule=='support_only' and a in ('blue_agent_1','blue_agent_3'))
                    is_restore = inputs[a].labels[i].startswith('Restore ')
                    interval = int(rule.rsplit('_',1)[1]) if rule.startswith('restore_interval_') else 0
                    if disable or (is_restore and step-last_restore[a]<interval):
                        actions[a] = inputs[a].labels.index('Sleep')
                    elif is_restore:
                        restores[a] += 1
                        last_restore[a] = step
                inputs,reward,done,truncated = env.step(actions)
                phase = int(env.base.environment_controller.state.mission_phase)
                monitor.push(step,phase,snapshot(env.base,monitor.protected,step))
                observed_reward = sum(reward.values())/len(reward)
                measured = measure(env.base)
                control = env.base.environment_controller
                impact = calculator.calculate_reward({}, {a:v for a,v in control.action.items() if 'red' in a},control.observation,False,control.state)
                parts = dict(work_failure=-measured['failed_weight'],impact=float(impact),
                             intervention=float(env.base.get_rewards()['Blue']['action_cost']))
                if abs(sum(parts.values())-observed_reward)>1e-8:
                    raise AssertionError('Official reward decomposition did not close')
                for k,v in parts.items():
                    decomposition[k] += v
                total += observed_reward
                costs.append(measured['cost'])
                if all(done.get(a,False) or truncated.get(a,False) for a in actions):
                    break
            outcome = monitor.finalize()
            row = dict(rule=rule,seed=seed,return_=total,event=outcome['event_cost'],first_hit=outcome['first_hit'],
                       legacy_cost=sum(costs)/len(costs),restores=restores,transitions=step,
                       phases={p:x['violation'] for p,x in outcome['label']['phases'].items()},
                       reward_parts=decomposition,ot=outcome['measurement'])
            rows.append(row)
            write(args.output/'results.json',rows)
            write(args.output/'state.json',dict(state='running',pid=os.getpid(),identity=identity,completed=len(rows),total=len(plan),heartbeat=time.time()))
            print(json.dumps({k:v for k,v in row.items() if k!='ot'}),flush=True)
        summary = {r:dict(episodes=len(t:=[x for x in rows if x['rule']==r]),mean_return=sum(x['return_'] for x in t)/len(t),events=sum(x['event'] for x in t)) for r in rules}
        if args.mode=='confirmation':
            item = summary['alert_restore']
            item['one_sided_95_upper'] = upper_bound(item['events'],item['episodes'])
            item['risk_requirement_supported'] = item['one_sided_95_upper']<=.05
        write(args.output/'summary.json',summary)
        write(args.output/'state.json',dict(state='phase_complete',completed=len(rows),total=len(plan),pid=os.getpid(),identity=identity))
    except BaseException as error:
        write(args.output/'state.json',dict(state='stopped' if isinstance(error,(InterruptedError,KeyboardInterrupt)) else 'failed',reason=str(error),completed=len(rows),pid=os.getpid(),identity=identity))
        raise


if __name__=='__main__':
    main()
