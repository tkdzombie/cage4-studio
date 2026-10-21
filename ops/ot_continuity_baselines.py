"""Bounded, separately frozen OT baseline smoke/pilot tests; never resumes old runs."""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value));tmp.replace(path)


def finite_tree(value):
    import torch
    if torch.is_tensor(value):return bool(torch.isfinite(value).all())
    if isinstance(value,dict):return all(finite_tree(v) for v in value.values())
    if isinstance(value,(list,tuple)):return all(finite_tree(v) for v in value)
    if isinstance(value,float):return math.isfinite(value)
    return True


def feasibility_admission(path,runtime,contract):
    """The learning screen requires complete, independently checked evidence."""
    from ot_task_confirmation import upper_bound
    from cage4_baselines.safety.continuity_contract import evaluate
    path=Path(path).expanduser().resolve()
    state=json.loads((path/'state.json').read_text())
    manifest=json.loads((path/'manifest.json').read_text())
    rows=json.loads((path/'results.json').read_text())
    source=Path(__file__).with_name('ot_patrol_probe.py')
    if state['state']!='phase_complete' or manifest.get('scope')!='confirmation' or manifest.get('alpha')!=.025:
        raise RuntimeError('Completed fixed-policy risk confirmation required')
    if manifest['runtime']!=runtime or hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['script_sha256']:
        raise RuntimeError('Feasibility source/runtime drift')
    expected=list(range(56001,56129))
    if manifest['plan']!=[['analyse',s] for s in expected] or [r['seed'] for r in rows]!=expected:
        raise RuntimeError('Confirmation scenario plan incomplete or changed')
    for row in rows:
        if row['mode']!='analyse' or row['transitions']!=499 or type(row['event']) is not int or row['event'] not in (0,1) or not math.isfinite(row['return_']):
            raise RuntimeError('Invalid completed confirmation episode')
        if evaluate(row['ot'],contract)['episode_violation'] is not bool(row['event']):
            raise RuntimeError('Confirmation measurement/event mismatch')
    events=sum(r['event'] for r in rows);upper=upper_bound(events,len(rows),.025)
    if upper>.05:raise RuntimeError('Risk evidence does not support the fixed 5% requirement')
    return dict(control=str(path),episodes=len(rows),events=events,one_sided_97_5_upper=upper,
                note='Feasible rule witness only, not a learning-policy safety certificate')


def admission(preflight,runtime,contract_sha,upstreams):
    import ast
    import torch
    preflight=Path(preflight).expanduser().resolve()
    state=json.loads((preflight/'state.json').read_text())
    if state['state']!='phase_complete' or state['scope']!='integration':
        raise RuntimeError('Completed integration evidence is required')
    if {r['algorithm'] for r in state['results']}!={'macpo','mappo_lagrangian'}:
        raise RuntimeError('Both constrained baseline integrations are required')
    core=Path(upstreams).expanduser()/'macpo/MACPO/macpo/algorithms/r_mappo/r_macpo.py'
    text=core.read_text();tree=ast.parse(text)
    method=next(node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef) and node.name=='trpo_update')
    method_source=ast.get_source_segment(text,method)
    if method_source.count('self.gamma')!=1 or '(1 - self.gamma)' not in method_source:
        raise RuntimeError('Pinned CPO rescaling assumption changed')
    checked=[]
    for result in state['results']:
        checkpoint=Path(result['checkpoint']);payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if not finite_tree(payload) or payload['metadata']['runtime']!=runtime or payload['metadata']['contract_sha256']!=contract_sha:
            raise RuntimeError('Preflight checkpoint/source/contract validation failed')
        if payload['metadata']['event_budget']!=.05 or not result['checkpoint_roundtrip'] or not result['evaluation_weights_unchanged']:
            raise RuntimeError('Budget or weight-isolation gate failed')
        folder=Path(result['run']);updates=json.loads((folder/'updates.json').read_text())
        if not updates or not finite_tree(updates):raise RuntimeError('No finite real optimizer update')
        metrics=updates[-1]['metrics']
        if not all(m['event_budget']==.05 and m['actor_changed'] and m['critic_changed'] and m['cost_critic_changed'] for m in metrics.values()):
            raise RuntimeError('Real actor and both critic updates required')
        evaluations=result['evaluations']
        if len(evaluations)!=4 or {(e['mode'],e['seed']) for e in evaluations}!={(m,s) for m in ('stochastic','greedy') for s in (54001,54002)}:
            raise RuntimeError('Complete dual-mode development check required')
        checked.append(dict(algorithm=result['algorithm'],checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),optimizer_states_finite=True))
    return dict(training_enabled_for_bounded_pilot=True,preflight_control=str(preflight),
                runtime=runtime,contract_sha256=contract_sha,checkpoints=checked,
                cpo_method_sha256=hashlib.sha256(method_source.encode()).hexdigest(),
                gate='engineering acceptance only; feasibility and safety not established')


def validate_mechanism_plan(plan):
    expected={'mappo_lagrangian':{'lamda_lagr':100.},'macpo':{'fraction_coef':1.}}
    if plan.get('schema')!='cage4.ot-mechanism-candidates/1':raise ValueError('Unknown candidate plan')
    if (plan.get('training_seed'),plan.get('episodes'),plan.get('batch'))!=(101,256,16):raise ValueError('Fixed 256/16/101 comparison required')
    if plan.get('evaluation_seeds')!=list(range(54001,54021)):raise ValueError('Fixed development scenes required')
    if plan.get('max_candidates')!=2 or plan.get('max_seconds_per_task')!=21600 or plan.get('minimum_free_gib')!=30:raise ValueError('Candidate/resource bounds changed')
    candidates=plan.get('candidates',{})
    if set(candidates)!=set(expected):raise ValueError('Exactly two named candidates required')
    for name,overrides in expected.items():
        item=candidates[name]
        if any(type(v) not in (int,float) for v in item.get('hyperparameters',{}).values()):raise ValueError('Numeric candidate values required')
        if item.get('hyperparameters')!=overrides or not item.get('question'):raise ValueError('Only the preregistered single-variable comparisons are permitted')
    return plan


def mechanism_admission(plan_path,runtime,contract_sha):
    """Require the completed unchanged baseline and matched frozen diagnostic."""
    plan_path=Path(plan_path).expanduser().resolve()
    plan=validate_mechanism_plan(json.loads(plan_path.read_text()))
    baseline=Path(plan['baseline_control']).expanduser().resolve()
    diagnostic=Path(plan['diagnostic_control']).expanduser().resolve()
    state=json.loads((baseline/'state.json').read_text())
    ds=json.loads((diagnostic/'state.json').read_text())
    dp=json.loads((diagnostic/'plan.json').read_text())
    if state['state']!='phase_complete' or state['scope']!='learning-screen' or ds!={'state':'phase_complete','environment_transitions':3992,'parameter_updates':0}:
        raise RuntimeError('Completed baseline and bounded diagnostic required')
    if hashlib.sha256((baseline/'contract.json').read_bytes()).hexdigest()!=contract_sha or dp['runtime']!=runtime or dp['contract_sha256']!=contract_sha:
        raise RuntimeError('Basis source or contract drift')
    checked=[]
    for algorithm in ('mappo_lagrangian','macpo'):
        original=next(r for r in state['results'] if r['algorithm']==algorithm)
        run=Path(original['run']);manifest=json.loads((run/'manifest.json').read_text())
        evidence=json.loads((diagnostic/(algorithm+'.result.json')).read_text())
        if manifest['runtime']!=runtime or manifest['config']['seed']!=101 or manifest['config']['episodes']!=256 or manifest['config']['rollout_episodes']!=16 or manifest['event_budget']!=.05:
            raise RuntimeError('Fixed baseline comparison changed')
        if not original['checkpoint_roundtrip'] or not original['evaluation_weights_unchanged'] or original['training_episodes']!=256 or original['updates']!=16:
            raise RuntimeError('Baseline engineering result incomplete')
        expected={(mode,seed) for mode in ('stochastic','greedy') for seed in range(54001,54021)}
        if {(r['mode'],r['seed']) for r in original['evaluations']}!=expected or len(original['evaluations'])!=40:raise RuntimeError('Both baseline modes required')
        if evidence['runtime']!=runtime or evidence['contract_sha256']!=contract_sha or evidence['environment_transitions']!=1996 or evidence['parameter_updates']!=0 or not evidence['all_weights_unchanged'] or not evidence['checkpoint_and_optimizers_finite'] or not finite_tree(evidence):
            raise RuntimeError('Invalid matched diagnostic')
        checkpoint_sha=hashlib.sha256(Path(original['checkpoint']).read_bytes()).hexdigest()
        if evidence['checkpoint_sha256']!=checkpoint_sha or [r['seed'] for r in evidence['episodes']]!=list(range(54001,54005)):
            raise RuntimeError('Diagnostic policy or scene drift')
        for row in evidence['episodes']:
            ref=next(r for r in original['evaluations'] if r['mode']=='stochastic' and r['seed']==row['seed'])
            if not row['matches_frozen_evaluation'] or (row['return_'],row['event'],row['first_hit'])!=(ref['return_'],ref['event_cost'],ref['first_hit']):raise RuntimeError('Diagnostic replay mismatch')
        checked.append(dict(algorithm=algorithm,checkpoint_sha256=checkpoint_sha))
    return plan,dict(plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(),baseline_control=str(baseline),diagnostic_control=str(diagnostic),checkpoints=checked,
                     scope='Exactly two bounded mechanism candidates, not final configuration selection or qualification')


def reward_value_diagnostics(policy,history,reward_scale):
    import numpy as np
    import torch
    from cage4_baselines.safety.continuity_events import undiscounted_returns
    T=len(history);terminal=[h[3] for h in history]
    central=np.concatenate([h[0][policy.names[0]]['cent'] for h in history])
    zero=np.zeros((T,policy.args.recurrent_N,policy.args.hidden_size),np.float32)
    masks=np.ones((T,1),np.float32);result={}
    for a in policy.names:
        target=np.asarray(undiscounted_returns([h[1][a]*reward_scale for h in history],terminal),np.float32)
        before=np.concatenate([h[0][a]['value'] for h in history]).reshape(-1)
        with torch.no_grad():after=policy.policies[a].get_values(central,zero,masks).numpy().reshape(-1)
        result[a]=dict(reward_mc_mse_before=float(np.mean((target-before)**2)),reward_mc_mse_after=float(np.mean((target-after)**2)),
                       reward_mc_bias_before=float(np.mean(target-before)),reward_mc_bias_after=float(np.mean(target-after)),
                       reward_target_variance=float(np.var(target)),reward_negative_tail_before_fraction=float(np.mean(target-before < -policy.trainers[a].huber_delta)))
    return result


def worker(task_path):
    task=json.loads(Path(task_path).read_text());control=Path(task['control'])
    sys.path.insert(0,str(control/'source'))
    import numpy as np
    import torch
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.finite_horizon import FiniteHorizonBaseline
    from cage4_baselines.paper.backend import fingerprint
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.service_audit import inventory,snapshot
    from cage4_baselines.safety.continuity_events import ContinuityEvents
    from cage4_baselines.safety.costs import measure
    from cage4_protocol.store import process_identity
    contract=json.loads((control/'contract.json').read_text());contract_sha=hashlib.sha256((control/'contract.json').read_bytes()).hexdigest()
    c=task['config'];algorithm=c['algorithm'];mechanism_plan=None
    if task['scope']=='mechanism-screen':
        plan_path=control/'mechanism-plan.json';mechanism_plan=validate_mechanism_plan(json.loads(plan_path.read_text()))
        if hashlib.sha256(plan_path.read_bytes()).hexdigest()!=task['mechanism_plan_sha256'] or any(c['hyperparameters'].get(k)!=v for k,v in mechanism_plan['candidates'][algorithm]['hyperparameters'].items()):raise RuntimeError('Frozen candidate plan/config mismatch')
    folder=Path(task['run']);folder.mkdir(exist_ok=False)
    def identities():return {a:{k:fingerprint(getattr(p,k)) for k in ('actor','critic','cost_critic')} for a,p in policy.policies.items()}
    random.seed(c['seed']);np.random.seed(c['seed']);torch.manual_seed(c['seed']);torch.set_num_threads(c['torch_threads'])
    env=Cage4Adapter(c['seed'],500,observation=c['observation'],include_graph=False)
    risk=contract['risk_limit']['numerator']/contract['risk_limit']['denominator']
    policy=FiniteHorizonBaseline(env.reset(),c,task['upstreams'],risk)
    runtime=runtime_identity()
    if runtime!=task['runtime']:raise RuntimeError('Frozen runtime mismatch')
    metadata=dict(schema='cage4.ot-baseline-test/1',variant=task.get('variant',algorithm+'-ot-finite-horizon-v1'),config=c,
                  contract_sha256=contract_sha,runtime=runtime,upstream=policy.provenance,
                  pid=os.getpid(),identity=process_identity(os.getpid()),training_enabled=True,
                  reward_returns='undiscounted MC, scale '+str(c['reward_scale']),cost_returns='undiscounted first-hit MC',
                  event_budget=risk,actor_information='own legal Blue vector only',
                  critic_information='concatenated legal Blue vectors only',scope=task['scope'])
    if task.get('mechanism_plan_sha256'):metadata['mechanism_plan_sha256']=task['mechanism_plan_sha256']
    if task.get('stream_training'):metadata['training_record_format']='JSONL each episode; JSON snapshot at batch updates'
    if task.get('reward_diagnostics'):metadata['reward_critic_diagnostics']=True
    metadata['runner_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    metadata['evaluation_seeds']=task.get('evaluation_seeds',[54001,54002])
    write(folder/'manifest.json',metadata);write(folder/'status.json',dict(state='running',**metadata))
    records=[];updates=[];train_seeds=[100000+c['seed']*100000+i for i in range(c['episodes'])]
    def collect(seed,training,greedy=False):
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        inputs=env.reset(seed);policy.begin_episode();monitor=ContinuityEvents(inventory(env.base),contract)
        total=0.;legacy=[]
        for step in range(1,501):
            if (control/'STOP').exists():raise InterruptedError('STOP requested')
            if step==1 or step%50==0:
                if process_identity(task['supervisor_pid'])!=task['supervisor_identity']:
                    raise InterruptedError('Supervisor identity disappeared')
            actions=policy.act(inputs,training=training,deterministic=greedy)
            inputs,rewards,done,truncated=env.step(actions)
            phase=int(env.base.environment_controller.state.mission_phase)
            cost=monitor.push(step,phase,snapshot(env.base,monitor.protected,step))
            terminal=not inputs or all(done.get(a,False) or truncated.get(a,False) for a in policy.names)
            if training:policy.observe(rewards,cost,terminal)
            total+=float(np.mean(list(rewards.values())));legacy.append(measure(env.base)['cost'])
            if step%50==0:write(folder/'progress.json',dict(pid=os.getpid(),identity=metadata['identity'],seed=seed,step=step,training=training,completed_training=len(records),heartbeat=time.time()))
            if terminal:break
        result=monitor.finalize()
        return dict(seed=seed,steps=step,return_=total,event_cost=result['event_cost'],first_hit=result['first_hit'],
                    legacy_availability_mean=float(np.mean(legacy)),ot=result['measurement'],mode='train' if training else ('greedy' if greedy else 'stochastic'))
    checkpoint=None
    def save_checkpoint(episode):
        nonlocal checkpoint
        checkpoint=folder/f'episode_{episode:06d}.pt'
        payload=dict(schema='cage4.ot-checkpoint/1',metadata=metadata,episode=episode,training_seeds=train_seeds[:episode],policy=policy.checkpoint())
        if not finite_tree(payload):raise FloatingPointError('Nonfinite model/optimizer checkpoint')
        torch.save(payload,checkpoint)
    try:
        for episode,seed in enumerate(train_seeds,1):
            row=collect(seed,True);records.append(row)
            if task.get('stream_training'):
                with (folder/'training.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                if episode%c['rollout_episodes']==0 or episode==c['episodes']:write(folder/'training.json',records)
            else:write(folder/'training.json',records)
            if mechanism_plan is not None and episode==16:
                basis=json.loads((Path(mechanism_plan['baseline_control'])/'state.json').read_text())
                original=next(r for r in basis['results'] if r['algorithm']==algorithm)
                if records!=json.loads((Path(original['run'])/'training.json').read_text())[:16]:raise RuntimeError('Single-variable candidate did not reproduce initial frozen-policy batch')
                write(folder/'first-batch-equivalence.json',dict(episodes=16,exact_rows_match=True))
            if episode%c['rollout_episodes']==0 or episode==c['episodes']:
                history=policy.history
                metrics=policy.update(c['reward_scale'])
                if task.get('reward_diagnostics'):
                    for a,extra in reward_value_diagnostics(policy,history,c['reward_scale']).items():metrics[a].update(extra)
                del history
                if not finite_tree(metrics):raise FloatingPointError('Nonfinite extended optimizer diagnostics')
                updates.append(dict(episode=episode,metrics=metrics));write(folder/'updates.json',updates)
            if episode%50==0:save_checkpoint(episode)
        save_checkpoint(c['episodes'])
        before=identities();loaded=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if loaded['metadata']!=metadata or loaded['training_seeds']!=train_seeds:raise RuntimeError('Checkpoint source/config/seed mismatch')
        policy.load(loaded['policy'])
        if identities()!=before:raise RuntimeError('Checkpoint roundtrip changed weights')
        evaluations=[]
        for greedy in (False,True):
            for seed in metadata['evaluation_seeds']:
                if seed in train_seeds:raise ValueError('Train/development overlap')
                evaluations.append(collect(seed,False,greedy))
                if identities()!=before:raise RuntimeError('Evaluation changed model weights')
                write(folder/'evaluations.json',evaluations)
        result=dict(algorithm=algorithm,variant=metadata['variant'],run=str(folder),training_episodes=len(records),
                    training_event_rate=sum(x['event_cost'] for x in records)/len(records),updates=len(updates),
                    evaluations=[{k:v for k,v in x.items() if k!='ot'} for x in evaluations],checkpoint=str(checkpoint),
                    all_weights_finite=True,checkpoint_roundtrip=True,evaluation_weights_unchanged=True)
        write(folder/'result.json',result);write(folder/'status.json',dict(state='finished',pid=os.getpid(),identity=metadata['identity']))
        write(task['result'],result)
    except BaseException as error:
        write(folder/'status.json',dict(state='stopped' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',pid=os.getpid(),identity=metadata['identity'],reason=str(error)))
        raise


def supervise(args):
    project=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(project/'protocol'),str(project/'cage4-baselines/src')]
    from cage4_protocol.store import process_identity,run_status
    from cage4_baselines.paper.__main__ import load_config
    from cage4_baselines.paper.provenance import runtime_identity
    root=args.control.expanduser().resolve();root.parent.mkdir(parents=True,exist_ok=True)
    runs=args.runs.expanduser().resolve();runs.mkdir(parents=True,exist_ok=True)
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    with (root.parent/'.constrained.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if shutil.disk_usage(runs).free<34*1024**3:raise RuntimeError('Insufficient reserve plus checkpoint margin')
        legacy=Path.home()/'cage4-experiments'
        for folder in legacy.iterdir():
            if folder.is_dir() and (folder/'manifest.json').exists() and run_status(folder).get('state') in ('running','unverified'):
                raise RuntimeError('Existing training activity must be resolved first')
        for folder in runs.iterdir():
            if folder.is_dir() and (folder/'manifest.json').exists() and run_status(folder).get('state') in ('running','unverified'):
                raise RuntimeError('Existing OT task must be resolved first')
        runtime=runtime_identity()
        contract_sha=hashlib.sha256((project/'cage4-baselines/configs/safety/ot-continuity-v1.json').read_bytes()).hexdigest()
        review=admission(args.preflight_control,runtime,contract_sha,args.upstreams) if args.scope!='integration' else dict(scope='integration_only')
        mechanism_plan=None
        if args.scope=='mechanism-screen':
            mechanism_plan,review['mechanism']=mechanism_admission(args.mechanism_plan,runtime,contract_sha)
        if args.scope in ('learning-screen','mechanism-screen'):
            review['feasibility']=feasibility_admission(args.feasibility_control,runtime,json.loads((project/'cage4-baselines/configs/safety/ot-continuity-v1.json').read_text()))
        root.mkdir(mode=0o700)
        source=root/'source';source.mkdir()
        for path,name in [(project/'cage4-baselines/src/cage4_baselines','cage4_baselines'),(project/'protocol/cage4_protocol','cage4_protocol')]:
            shutil.copytree(path,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(__file__,root/'runner.py')
        shutil.copyfile(project/'cage4-baselines/configs/safety/ot-continuity-v1.json',root/'contract.json')
        write(root/'admission.json',review)
        if mechanism_plan is not None:shutil.copyfile(args.mechanism_plan,root/'mechanism-plan.json')
        state=dict(schema='cage4.ot-campaign/1',state='running',pid=os.getpid(),identity=process_identity(os.getpid()),started_at=time.time(),scope=args.scope,results=[])
        def save(**values):
            state.update(values,heartbeat=time.time(),free_gib=shutil.disk_usage(runs).free/1024**3);write(root/'state.json',state)
        save()
        try:
            for algorithm in ('mappo_lagrangian','macpo'):
                if stopped or (root/'STOP').exists():raise InterruptedError('STOP before task')
                c=load_config(project/'cage4-baselines/configs/paper'/(algorithm+'.yaml'))
                c.update(episodes=args.episodes,rollout_episodes=args.batch,num_envs=1,record_replay=False)
                c['seed']=args.training_seed
                if mechanism_plan is not None:c['hyperparameters'].update(mechanism_plan['candidates'][algorithm]['hyperparameters'])
                folder=runs/(root.name+'-'+algorithm)
                result=root/(algorithm+'.result.json');task=root/(algorithm+'.task.json')
                write(task,dict(control=str(root),config=c,scope=args.scope,runtime=runtime,upstreams=str(args.upstreams.expanduser()),run=str(folder),result=str(result),supervisor_pid=os.getpid(),supervisor_identity=state['identity'],evaluation_seeds=list(range(54001,54021)) if args.scope in ('learning-screen','mechanism-screen') else [54001,54002],mechanism_plan_sha256=review.get('mechanism',{}).get('plan_sha256')))
                with (root/(algorithm+'.log')).open('w') as log:
                    child=subprocess.Popen([sys.executable,str(root/'runner.py'),'--task',str(task)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    identity=process_identity(child.pid);started=time.monotonic();sent=None
                    save(algorithm=algorithm,child_pid=child.pid,child_identity=identity,current_run=str(folder))
                    while child.poll() is None:
                        if (root/'STOP').exists() or shutil.disk_usage(runs).free<30*1024**3 or time.monotonic()-started>6*3600:stopped=True
                        if stopped and sent is None and process_identity(child.pid)==identity:
                            child.send_signal(signal.SIGINT);sent=time.monotonic()
                        if sent is not None and time.monotonic()-sent>120 and process_identity(child.pid)==identity:os.killpg(child.pid,signal.SIGTERM)
                        save();time.sleep(5)
                    if stopped:raise InterruptedError('Campaign stopped')
                    if child.returncode:raise RuntimeError(algorithm+' failed; inspect its log')
                state['results'].append(json.loads(result.read_text()));save()
            save(state='phase_complete',child_pid=None,child_identity=None)
        except BaseException as error:
            save(state='stopped' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',reason=str(error))
            raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path)
    p.add_argument('--control',type=Path)
    p.add_argument('--runs',type=Path,default=Path.home()/'cage4-experiments/ot-continuity-v1')
    p.add_argument('--upstreams',type=Path,default=Path.home()/'cage4-upstreams')
    p.add_argument('--episodes',type=int,default=32)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--scope',choices=('integration','bounded-pilot','learning-screen','mechanism-screen'),default='bounded-pilot')
    p.add_argument('--preflight-control',type=Path)
    p.add_argument('--feasibility-control',type=Path)
    p.add_argument('--mechanism-plan',type=Path)
    p.add_argument('--training-seed',type=int,choices=(101,102,103),default=101)
    a=p.parse_args()
    if a.task:worker(a.task)
    else:
        maximum=256 if a.scope in ('learning-screen','mechanism-screen') else 128
        if a.control is None or not 2<=a.episodes<=maximum or not 2<=a.batch<=16:p.error('Require new control and bounded episodes/batch')
        if a.scope!='integration' and a.preflight_control is None:p.error('Passed integration control required')
        if a.scope in ('learning-screen','mechanism-screen') and (a.feasibility_control is None or a.episodes!=256 or a.batch!=16):p.error('Learning screen requires passed risk witness, 256 episodes and batch 16')
        if a.scope=='mechanism-screen' and (a.mechanism_plan is None or a.training_seed!=101):p.error('Frozen mechanism plan and seed101 required')
        if a.scope!='mechanism-screen' and a.mechanism_plan is not None:p.error('Mechanism plan only applies to named candidate scope')
        if a.scope!='learning-screen' and a.training_seed!=101:p.error('Pilot/integration seed is fixed at 101')
        supervise(a)
