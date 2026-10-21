"""Two fixed public-information patrol witnesses; diagnostic data only."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import shutil
import time


class PublicPatrol:
    def __init__(self,mode,schemas):
        if mode not in ('analyse','restore'):raise ValueError(mode)
        self.mode,self.schemas,self.step=mode,schemas,0
        self.checked={a:{} for a in schemas};self.pending={a:{} for a in schemas}

    def act(self,inputs):
        self.step+=1;actions={}
        for a,value in inputs.items():
            legal={s:i for i,s in enumerate(value.labels) if value.mask[i]}
            schema=self.schemas[a];width=len(schema['hosts'][0]['fields'])
            rows=value.observation[len(schema['head']):].reshape(schema['max_hosts'],width)
            candidates=[]
            for item in schema['hosts']:
                host=item['hostname'];f=dict(zip(item['fields'],rows[item['slot']]))
                alert=bool(f['process_history'] or f['connection_history'] or (f['analysed'] and f['file_count']>0))
                if alert:self.pending[a].setdefault(host,self.step)
                else:self.pending[a].pop(host,None)
                if 'Restore '+host in legal and ('_user_host_' in host or '_server_host_' in host):candidates.append(host)
            alerts=[h for h in candidates if h in self.pending[a]]
            if alerts:
                host=min(alerts,key=lambda h:(self.pending[a][h],h));label='Restore '+host
            elif candidates:
                host=min(candidates,key=lambda h:(self.checked[a].get(h,-1),h))
                label=('Analyse ' if self.mode=='analyse' else 'Restore ')+host
            else:label='Sleep'
            if label not in legal:label='Sleep'
            actions[a]=legal[label]
            if label.startswith(('Restore ','Analyse ')):self.checked[a][label.split(' ',1)[1]]=self.step
        return actions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--expected-runtime',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scope',choices=('diagnostic','confirmation'),default='diagnostic')
    p.add_argument('--risk-module',type=Path)
    args=p.parse_args()
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.service_audit import inventory,snapshot
    from cage4_baselines.safety.continuity_events import ContinuityEvents
    from cage4_baselines.safety.costs import measure
    from CybORG.Shared.BlueRewardMachine import BlueRewardMachine
    from cage4_protocol.store import process_identity
    runtime=runtime_identity()
    assert runtime==json.loads(args.expected_runtime.read_text())['runtime']
    contract=json.loads(args.contract.read_text())
    assert hashlib.sha256(args.contract.read_bytes()).hexdigest()=='542d70e17de17625f821ae37ace340ff4f63697c9c9113bb48ac44d144b34248'
    if args.scope=='confirmation' and args.risk_module is None:raise ValueError('Exact risk-bound module required')
    seeds=list(range(56001,56129)) if args.scope=='confirmation' else [54001,54002,54003,54004,55003,55021,55026]
    plan=[(mode,seed) for mode in (('analyse',) if args.scope=='confirmation' else ('analyse','restore')) for seed in seeds]
    args.output.mkdir(exist_ok=False)
    identity=process_identity(os.getpid())
    def state(status,completed):
        temp=args.output/'state.tmp';temp.write_text(json.dumps(dict(state=status,completed=completed,total=len(plan),pid=os.getpid(),identity=identity,heartbeat=time.time())));temp.replace(args.output/'state.json')
    (args.output/'manifest.json').write_text(json.dumps(dict(plan=plan,max_transitions=len(plan)*499,
        training_updates=0,runtime=runtime,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        policy_sha256=hashlib.sha256(inspect.getsource(PublicPatrol).encode()).hexdigest(),
        scope=args.scope,alpha=.025 if args.scope=='confirmation' else None,
        formal_policy_family='Two fixed-policy confirmation attempts total; Bonferroni alpha .025 each. No additional attempts within this protocol.',
        selection='Analyse patrol selected on four development and three revealed failure cases, before fresh 128-case confirmation',
        pid=os.getpid(),identity=identity,
        information='PolicyInput/schema only; no health field or hidden OT counter')))
    state('running',0)
    env=Cage4Adapter(55001,500,include_graph=False);rows=[];started=time.monotonic()
    for mode,seed in plan:
        inputs=env.reset(seed);policy=PublicPatrol(mode,env.observation_schema())
        monitor=ContinuityEvents(inventory(env.base),contract)
        total=0.;parts=dict(work_failure=0.,impact=0.,intervention=0.)
        calculator=BlueRewardMachine('Blue');counts={}
        for step in range(1,500):
            if (args.output/'STOP').exists() or time.monotonic()-started>3600 or shutil.disk_usage(args.output).free<30*1024**3:
                raise InterruptedError('STOP/resource boundary')
            selected=policy.act(inputs)
            for a,i in selected.items():
                command=inputs[a].labels[i].split()[0];counts[command]=counts.get(command,0)+1
            inputs,reward,done,truncated=env.step(selected)
            control=env.base.environment_controller
            monitor.push(step,int(control.state.mission_phase),snapshot(env.base,monitor.protected,step))
            measured=measure(env.base)
            impact=calculator.calculate_reward({}, {a:v for a,v in control.action.items() if 'red' in a},control.observation,False,control.state)
            current=dict(work_failure=-measured['failed_weight'],impact=float(impact),intervention=float(env.base.get_rewards()['Blue']['action_cost']))
            r=sum(reward.values())/len(reward)
            assert abs(sum(current.values())-r)<1e-8
            total+=r
            for k,v in current.items():parts[k]+=v
            if all(done.get(a,False) or truncated.get(a,False) for a in selected):break
        outcome=monitor.finalize()
        row=dict(mode=mode,seed=seed,return_=total,event=outcome['event_cost'],first_hit=outcome['first_hit'],
                 reward_parts=parts,actions=counts,transitions=step,ot=outcome['measurement'])
        rows.append(row);temp=args.output/'results.tmp';temp.write_text(json.dumps(rows));temp.replace(args.output/'results.json')
        state('running',len(rows))
        print(json.dumps({k:v for k,v in row.items() if k!='ot'}),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(phase_complete=True,episodes=len(rows),transitions=sum(r['transitions'] for r in rows))))
    if args.scope=='confirmation':
        spec=importlib.util.spec_from_file_location('risk_bounds',args.risk_module)
        bound=importlib.util.module_from_spec(spec);spec.loader.exec_module(bound)
        k=sum(r['event'] for r in rows);upper=bound.upper_bound(k,len(rows),.025)
        (args.output/'summary.json').write_text(json.dumps(dict(episodes=len(rows),events=k,mean_return=sum(r['return_'] for r in rows)/len(rows),
            one_sided_97_5_upper=upper,risk_requirement_supported=upper<=.05,
            inference='Per-policy conditional iid scenario bound; two-attempt Bonferroni family at least 95% simultaneous coverage')))
    state('phase_complete',len(rows))


if __name__=='__main__':main()
