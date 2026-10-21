"""Exactly replay all failures of a completed fixed-rule confirmation.

No policy changes: hidden OT state labels the diagnosis after action selection.
The already revealed confirmation becomes diagnostic evidence, never reused as
independent validation for any improved rule.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation',type=Path,required=True)
    p.add_argument('--base-probe',type=Path,required=True)
    p.add_argument('--contract',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.continuity_events import ContinuityEvents
    from cage4_baselines.safety.service_audit import inventory,snapshot
    original=json.loads((args.confirmation/'manifest.json').read_text())
    assert json.loads((args.confirmation/'state.json').read_text())['state']=='phase_complete'
    assert runtime_identity()==original['runtime']
    assert hashlib.sha256(args.base_probe.read_bytes()).hexdigest()==original['base_sha256']
    assert hashlib.sha256(args.contract.read_bytes()).hexdigest()==original['contract_sha256']
    cases=[x for x in json.loads((args.confirmation/'results.json').read_text()) if x['event']]
    if len(cases)>6:
        raise ValueError('Replay cap is six cases')
    args.output.mkdir(exist_ok=False)
    (args.output/'manifest.json').write_text(json.dumps(dict(source_confirmation=str(args.confirmation),
        cases=[{'seed':x['seed'],'return_':x['return_'],'first_hit':x['first_hit']} for x in cases],
        max_transitions=len(cases)*499,training_updates=0,runtime=original['runtime'],
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())))
    spec=importlib.util.spec_from_file_location('frozen_probe',args.base_probe)
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    env=Cage4Adapter(original['plan'][0][1],500,include_graph=False)
    contract=json.loads(args.contract.read_text());started=time.monotonic();results=[]
    for case in cases:
        inputs=env.reset(case['seed']);schemas=env.observation_schema()
        controller=probe.PublicRule('alert_restore',schemas)
        monitor=ContinuityEvents(inventory(env.base),contract)
        trace=[];total=0.;runs={};hits={}
        for step in range(1,500):
            if (args.output/'STOP').exists() or time.monotonic()-started>3600 or shutil.disk_usage(args.output).free<30*1024**3:
                raise InterruptedError('STOP/resource boundary')
            actions=controller.act(inputs)
            view={}
            for a,value in inputs.items():
                schema=schemas[a];width=len(schema['hosts'][0]['fields'])
                rows=value.observation[len(schema['head']):].reshape(schema['max_hosts'],width)
                hosts={}
                for h in schema['hosts']:
                    fields=dict(zip(h['fields'],rows[h['slot']]))
                    hosts[h['hostname']]={k:float(fields[k]) for k in ('process_history','connection_history','alert_age','action_age','action_success')}
                view[a]=dict(action=value.labels[actions[a]],busy_only_sleep=int(value.mask.sum())==1,hosts=hosts)
            trace.append(dict(step=step,public=view))
            inputs,reward,done,truncated=env.step(actions)
            phase=int(env.base.environment_controller.state.mission_phase)
            samples=snapshot(env.base,monitor.protected,step)
            monitor.push(step,phase,samples)
            for row in samples:
                key=(phase,row['host']);bad=not row['service_present'] or not row['active']
                runs[key]=runs.get(key,0)+1 if bad else 0
                if runs[key]==17 and key not in hits:
                    hits[key]=step
            total+=sum(reward.values())/len(reward)
            if all(done.get(a,False) or truncated.get(a,False) for a in actions):break
        outcome=monitor.finalize()
        assert total==case['return_'] and outcome['event_cost']==case['event'] and outcome['first_hit']==case['first_hit'], 'Replay mismatch'
        summaries=[]
        for (phase,host),hit in hits.items():
            owner=next(a for a,schema in schemas.items() if any(h['hostname']==host for h in schema['hosts']))
            window=[r for r in trace if hit-16<=r['step']<=hit]
            summaries.append(dict(phase=phase,host=host,first_crossing=hit,owner=owner,
                legal_alert_steps=[r['step'] for r in window if any(r['public'][owner]['hosts'][host][k] for k in ('process_history','connection_history'))],
                restores_this_host=[r['step'] for r in window if r['public'][owner]['action']=='Restore '+host],
                restores_other_hosts=[dict(step=r['step'],action=r['public'][owner]['action']) for r in window if r['public'][owner]['action'].startswith('Restore ') and r['public'][owner]['action']!='Restore '+host],
                busy_steps=[r['step'] for r in window if r['public'][owner]['busy_only_sleep']]))
        row=dict(seed=case['seed'],return_=total,event=outcome['event_cost'],exact_replay=True,failures=summaries)
        results.append(row)
        (args.output/f"trace-{case['seed']}.json").write_text(json.dumps(trace))
        (args.output/'results.json').write_text(json.dumps(results))
        print(json.dumps(row),flush=True)
    (args.output/'complete.json').write_text(json.dumps(dict(phase_complete=True,episodes=len(results),transitions=len(results)*499)))


if __name__=='__main__':main()
