"""Bounded MACPO critic-only experiment on discarded model copies.

Collect four fixed development episodes from an existing checkpoint, verify
their recorded outcomes, then compare the author's value clipping option.
Original actor, critics and optimizers are never trained or replaced.
"""
import argparse
from copy import copy, deepcopy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

SEEDS=[54001,54002,54003,54004]
FIT_STEPS=20


def write(path,value):
    temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value));temp.replace(path)


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
    return h.hexdigest()


def fork_critic(model,optimizer,state):
    """Every arm owns its parameters AND Adam moment/step tensors."""
    cloned=deepcopy(model)
    cloned_optimizer=type(optimizer)(cloned.parameters(),**deepcopy(optimizer.defaults))
    cloned_optimizer.load_state_dict(deepcopy(state))
    return cloned,cloned_optimizer


def worker(args):
    basis=args.basis_control.resolve();out=args.output.resolve()
    sys.path.insert(0,str(basis/'source'))
    import numpy as np
    import torch
    from cage4_protocol.store import process_identity
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.finite_horizon import FiniteHorizonBaseline
    from cage4_baselines.paper.backend import fingerprint
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.service_audit import inventory,snapshot
    from cage4_baselines.safety.continuity_events import ContinuityEvents,undiscounted_returns
    plan=json.loads((out/'plan.json').read_text())
    assert digest(Path(__file__))==plan['script_sha256'] and plan['seeds']==SEEDS
    parent=plan['supervisor_pid'];parent_identity=plan['supervisor_identity']
    def check():
        if (out/'STOP').exists() or process_identity(parent)!=parent_identity:raise InterruptedError('STOP or supervisor changed')
    state=json.loads((basis/'state.json').read_text());assert state['state']=='phase_complete'
    result=next(r for r in state['results'] if r['algorithm']=='macpo')
    run=Path(result['run']);metadata=json.loads((run/'manifest.json').read_text());config=metadata['config']
    contract=json.loads((basis/'contract.json').read_text())
    assert runtime_identity()==metadata['runtime']==plan['runtime']
    assert digest(basis/'contract.json')==metadata['contract_sha256']==plan['contract_sha256']
    checkpoint=Path(result['checkpoint']);assert digest(checkpoint)==plan['checkpoint_sha256']
    spec=importlib.util.spec_from_file_location('basis_runner',basis/'runner.py')
    runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
    payload=torch.load(checkpoint,map_location='cpu',weights_only=True)
    assert runner.finite_tree(payload) and payload['metadata']==metadata and payload['episode']==256
    assert config['seed']==101 and metadata['event_budget']==.05
    reference={r['seed']:r for r in json.loads((run/'evaluations.json').read_text()) if r['mode']=='stochastic'}
    assert set(reference)==set(range(54001,54021))
    random.seed(101);np.random.seed(101);torch.manual_seed(101);torch.set_num_threads(2)
    env=Cage4Adapter(101,500,observation=config['observation'],include_graph=False)
    policy=FiniteHorizonBaseline(env.reset(),config,Path.home()/'cage4-upstreams',.05)
    assert policy.provenance==metadata['upstream']
    policy.load(payload['policy'])
    def fingerprints():return {a:{k:fingerprint(getattr(p,k)) for k in ('actor','critic','cost_critic')} for a,p in policy.policies.items()}
    before=fingerprints();episodes=[]
    for seed in SEEDS:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        inputs=env.reset(seed);policy.begin_episode();monitor=ContinuityEvents(inventory(env.base),contract);total=0.
        for step in range(1,500):
            if step==1 or step%50==0:check()
            actions=policy.act(inputs,training=True,deterministic=False)
            inputs,rewards,done,truncated=env.step(actions)
            phase=int(env.base.environment_controller.state.mission_phase)
            cost=monitor.push(step,phase,snapshot(env.base,monitor.protected,step))
            terminal=not inputs or all(done.get(a,False) or truncated.get(a,False) for a in policy.names)
            assert terminal==(step==499)
            policy.observe(rewards,cost,terminal);total+=float(np.mean(list(rewards.values())))
            if step%50==0:write(out/'progress.json',dict(stage='collect',seed=seed,step=step,transitions=len(policy.history),heartbeat=time.time()))
        label=monitor.finalize();ref=reference[seed]
        assert (total,label['event_cost'],label['first_hit'])==(ref['return_'],ref['event_cost'],ref['first_hit'])
        episodes.append(dict(seed=seed,return_=total,event=label['event_cost'],first_hit=label['first_hit'],exact_replay=True))
    history=policy.history;assert len(history)==1996 and sum(h[3] for h in history)==4
    central=np.concatenate([h[0][policy.names[0]]['cent'] for h in history])
    terminals=[h[3] for h in history]
    cost_target=np.asarray(undiscounted_returns([h[2] for h in history],terminals),np.float32)
    reward_target=np.stack([undiscounted_returns([h[1][a]*.01 for h in history],terminals) for a in policy.names],axis=1).astype(np.float32)
    predictions={key:np.stack([np.concatenate([h[0][a][key] for h in history]).reshape(-1) for a in policy.names],axis=1) for key in ('value','cost_value')}
    assert central.nbytes+reward_target.nbytes+cost_target.nbytes+sum(x.nbytes for x in predictions.values())<1024**3
    # Private cache prevents repeating simulator collection. Never warm-start a
    # policy with these development data or upload this artifact to the repo.
    np.savez_compressed(out/'legal-blue-critic-batch.npz',central=central,reward_target=reward_target,cost_target=cost_target,reward_prediction=predictions['value'],cost_prediction=predictions['cost_value'])
    observations=torch.from_numpy(central)
    zero=torch.zeros((1996,policy.args.recurrent_N,policy.args.hidden_size));masks=torch.ones((1996,1))
    def statistics(model,target):
        model.eval()
        with torch.no_grad():pred=model(observations,zero,masks)[0].reshape(-1)
        rows={}
        for name,section in (('fit',slice(0,998)),('heldout',slice(998,1996))):
            y=target[section];error=y-pred[section];variance=float(y.var(unbiased=False))
            rows[name]=dict(mse=float((error**2).mean()),bias=float(error.mean()),explained_variance=None if variance==0 else 1-float(error.var(unbiased=False))/variance)
        return rows
    rows=[];clone_steps=0
    for index,agent in enumerate(policy.names):
        original=policy.policies[agent];trainer=policy.trainers[agent]
        assert not trainer._use_popart
        for kind,module_name,optimizer_name,prediction_key in (('reward','critic','critic_optimizer','value'),('cost','cost_critic','cost_optimizer','cost_value')):
            target=torch.from_numpy(reward_target[:,index] if kind=='reward' else cost_target)
            old=torch.from_numpy(predictions[prediction_key][:,index,None])
            for clipped in (True,False):
                model,optimizer=fork_critic(getattr(original,module_name),getattr(original,optimizer_name),payload['policy'][agent]['optimizers'][optimizer_name])
                criterion=copy(trainer);criterion._use_clipped_value_loss=clipped
                with torch.no_grad():assert torch.max(torch.abs(model(observations,zero,masks)[0]-old))<1e-4
                series=[dict(step=0,**statistics(model,target))]
                for step in range(1,FIT_STEPS+1):
                    check();model.train();optimizer.zero_grad()
                    values=model(observations[:998],zero[:998],masks[:998])[0]
                    loss=criterion.cal_value_loss(values,old[:998],target[:998,None],masks[:998])
                    value_grad=torch.autograd.grad(loss,values,retain_graph=True)[0]
                    zero_fraction=float((value_grad.abs()<1e-12).float().mean())
                    (loss*criterion.value_loss_coef).backward()
                    if criterion._use_max_grad_norm:torch.nn.utils.clip_grad_norm_(model.parameters(),criterion.max_grad_norm)
                    optimizer.step();clone_steps+=1
                    if not runner.finite_tree(model.state_dict()) or not runner.finite_tree(optimizer.state_dict()):raise FloatingPointError('Nonfinite cloned critic')
                    if step in (1,5,10,20):series.append(dict(step=step,loss=float(loss.detach()),zero_value_gradient_fraction=zero_fraction,**statistics(model,target)))
                rows.append(dict(agent=agent,kind=kind,clipped_value_loss=clipped,series=series))
                write(out/'fit-progress.json',dict(completed_arms=len(rows),total_arms=20,clone_optimizer_steps=clone_steps,heartbeat=time.time()))
                del model,optimizer,criterion
    assert clone_steps==400 and fingerprints()==before
    aggregate={}
    for kind in ('reward','cost'):
        aggregate[kind]={}
        for clipped in (True,False):
            selected=[r['series'][-1] for r in rows if r['kind']==kind and r['clipped_value_loss']==clipped]
            aggregate[kind]['clipped' if clipped else 'unclipped']={part:float(np.mean([r[part]['mse'] for r in selected])) for part in ('fit','heldout')}
    improve=sum(next(r for r in rows if r['agent']==a and r['kind']=='reward' and not r['clipped_value_loss'])['series'][-1]['heldout']['mse']<next(r for r in rows if r['agent']==a and r['kind']=='reward' and r['clipped_value_loss'])['series'][-1]['heldout']['mse'] for a in policy.names)
    gate=(aggregate['reward']['unclipped']['heldout']<=.8*aggregate['reward']['clipped']['heldout'] and improve>=4 and aggregate['cost']['unclipped']['heldout']<=1.2*aggregate['cost']['clipped']['heldout'])
    write(out/'result.json',dict(schema='cage4.macpo-value-fit/1',runtime=metadata['runtime'],checkpoint_sha256=plan['checkpoint_sha256'],contract_sha256=plan['contract_sha256'],environment_transitions=1996,policy_parameter_updates=0,clone_optimizer_steps=400,original_policy_weights_unchanged=True,episodes=episodes,aggregate=aggregate,reward_agents_improved=improve,unclipped_candidate_gate=gate,rows=rows,note='Two fit and two heldout development scenes. Copies discarded; no policy training or risk qualification. Practical gate was fixed before collection; it is not a safety threshold.'))


def supervise(args):
    basis=args.basis_control.expanduser().resolve();out=args.output.expanduser().resolve()
    sys.path.insert(0,str(basis/'source'))
    from cage4_protocol.store import process_identity
    import fcntl
    state=json.loads((basis/'state.json').read_text());assert state['state']=='phase_complete'
    result=next(r for r in state['results'] if r['algorithm']=='macpo')
    metadata=json.loads((Path(result['run'])/'manifest.json').read_text())
    assert metadata['config']['seed']==101 and metadata['config']['algorithm']=='macpo'
    with (out.parent/'.constrained.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(out.parent).free>=31*1024**3
        out.mkdir(mode=0o700)
        shutil.copyfile(__file__,out/'runner.py')
        write(out/'plan.json',dict(schema='cage4.macpo-value-fit-plan/1',seeds=SEEDS,fit_seeds=SEEDS[:2],heldout_seeds=SEEDS[2:],max_environment_transitions=1996,policy_updates=0,max_cloned_critic_optimizer_steps=400,fit_steps=20,max_seconds=3600,min_free_gib=30,script_sha256=digest(out/'runner.py'),checkpoint_sha256=digest(Path(result['checkpoint'])),runtime=metadata['runtime'],contract_sha256=metadata['contract_sha256'],supervisor_pid=os.getpid(),supervisor_identity=process_identity(os.getpid()),candidate_gate='Heldout reward MSE <=0.8x clipped in mean and improved >=4/5 agents; cost MSE <=1.2x clipped. Diagnostic criterion, not safety requirement.'))
        started=time.monotonic()
        with (out/'worker.log').open('x') as log:
            child=subprocess.Popen([sys.executable,str(out/'runner.py'),'--basis-control',str(basis),'--output',str(out),'--worker'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            identity=process_identity(child.pid)
            write(out/'state.json',dict(state='running',pid=os.getpid(),identity=process_identity(os.getpid()),child_pid=child.pid,child_identity=identity))
            try:
                while child.poll() is None:
                    if (out/'STOP').exists() or shutil.disk_usage(out).free<30*1024**3 or time.monotonic()-started>3600:raise InterruptedError('STOP/resource boundary')
                    time.sleep(2)
                if child.returncode:raise RuntimeError('Critic-only worker failed; preserve log')
                write(out/'state.json',dict(state='phase_complete',environment_transitions=1996,policy_parameter_updates=0,cloned_critic_optimizer_steps=400))
            except BaseException as error:
                if child.poll() is None and process_identity(child.pid)==identity:
                    child.send_signal(signal.SIGINT)
                    try:child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        if process_identity(child.pid)==identity:os.killpg(child.pid,signal.SIGTERM)
                        child.wait(timeout=30)
                write(out/'state.json',dict(state='stopped' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',reason=str(error)))
                raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis-control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    worker(args) if args.worker else supervise(args)
