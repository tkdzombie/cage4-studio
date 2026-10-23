"""Only MACPO, only training seed101; bounded Huber-repair validation/training."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import os
from ot_continuity_baselines import write,finite_tree


def supervise(args):
    project=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(project/'protocol'),str(project/'cage4-baselines/src')]
    import torch
    from cage4_protocol.store import process_identity,run_status
    from cage4_baselines.paper.__main__ import load_config
    from cage4_baselines.paper.provenance import runtime_identity
    out=args.control.expanduser().resolve();repair=args.repair_control.expanduser().resolve();basis=args.basis_control.expanduser().resolve()
    runs=Path.home()/'cage4-experiments/ot-continuity-v1';upstreams=Path.home()/'cage4-upstreams'
    runtime=runtime_identity();evidence=json.loads((repair/'result.json').read_text())
    assert json.loads((repair/'state.json').read_text())['state']=='phase_complete' and evidence['runtime']==runtime
    assert evidence['environment_transitions']==0 and evidence['policy_parameter_updates']==0 and evidence['original_policy_weights_unchanged'] and finite_tree(evidence)
    basis_state=json.loads((basis/'state.json').read_text());assert basis_state['state']=='phase_complete'
    reference=next(r for r in basis_state['results'] if r['algorithm']=='macpo')
    reference_metadata=json.loads((Path(reference['run'])/'manifest.json').read_text())
    contract_sha=hashlib.sha256((project/'cage4-baselines/configs/safety/ot-continuity-v1.json').read_bytes()).hexdigest()
    assert contract_sha==reference_metadata['contract_sha256']=='542d70e17de17625f821ae37ace340ff4f63697c9c9113bb48ac44d144b34248'
    # The loss correction must be the only algorithm-source change.
    def sources(root):return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.py')}
    old=sources(basis/'source/cage4_baselines');new=sources(project/'cage4-baselines/src/cage4_baselines')
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    assert changed=={'paper/discrete_macpo.py','paper/value_loss.py'},changed
    config=load_config(project/'cage4-baselines/configs/paper/macpo.yaml')
    config.update(seed=101,episodes=2 if args.scope=='integration' else 1792,rollout_episodes=2 if args.scope=='integration' else 16,num_envs=1,record_replay=False)
    config['hyperparameters']['fraction_coef']=1.
    assert config['reward_scale']==.01 and config['steps']==500
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    with (out.parent/'.constrained.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for folder in runs.iterdir():
            if folder.is_dir() and (folder/'manifest.json').exists() and run_status(folder).get('state') in ('running','unverified'):raise RuntimeError('Resolve existing activity first')
        admission=dict(scope='MACPO-only, one training seed101',runtime=runtime,contract_sha256=contract_sha,repair_control=str(repair),changed_algorithm_sources=sorted(changed),original_comparison=str(basis),six_hour_limit=True)
        if args.scope=='learning':
            if args.preflight_control is None:raise ValueError('Fresh repaired MACPO integration required')
            pre=args.preflight_control.expanduser().resolve();prestate=json.loads((pre/'state.json').read_text())
            assert prestate['state']=='phase_complete' and prestate['scope']=='integration'
            r=prestate['result'];payload=torch.load(r['checkpoint'],map_location='cpu',weights_only=True)
            assert finite_tree(payload) and payload['metadata']['runtime']==runtime and payload['metadata']['event_budget']==.05 and payload['metadata']['contract_sha256']==contract_sha
            assert r['algorithm']=='macpo' and r['training_episodes']==2 and r['updates']==1 and r['checkpoint_roundtrip'] and r['evaluation_weights_unchanged']
            updates=json.loads((Path(r['run'])/'updates.json').read_text())
            assert any(m['actor_changed'] for m in updates[0]['metrics'].values())
            assert all(m['critic_changed'] and m['cost_critic_changed'] and 'reward_mc_mse_after' in m for m in updates[0]['metrics'].values())
            assert len(r['evaluations'])==4
            forecast=(config['episodes']//50+int(config['episodes']%50!=0))*Path(r['checkpoint']).stat().st_size*1.05/1024**3+1
            admission.update(preflight_control=str(pre),forecast_checkpoint_gib=forecast,episodes=1792,updates=112,scope_note='No multi-training-seed replication. Fixed development scenes still evaluate the one policy. Prior measured throughput ~5h19 training; six-hour hard stop includes evaluation. No automatic extension.')
        else:forecast=1.
        free=shutil.disk_usage(runs).free/1024**3
        if free<30+forecast:raise RuntimeError('Insufficient complete checkpoint reserve')
        out.mkdir(mode=0o700);source=out/'source';source.mkdir()
        for path,name in ((project/'cage4-baselines/src/cage4_baselines','cage4_baselines'),(project/'protocol/cage4_protocol','cage4_protocol')):shutil.copytree(path,source/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copyfile(project/'ops/ot_continuity_baselines.py',out/'runner.py');shutil.copyfile(__file__,out/'supervisor.py')
        shutil.copyfile(project/'cage4-baselines/configs/safety/ot-continuity-v1.json',out/'contract.json')
        write(out/'admission.json',admission)
        state=dict(schema='cage4.macpo-repair-campaign/1',state='running',scope=args.scope,pid=os.getpid(),identity=process_identity(os.getpid()),started_at=time.time())
        def save(**values):state.update(values,heartbeat=time.time(),free_gib=shutil.disk_usage(runs).free/1024**3);write(out/'state.json',state)
        save();folder=runs/(out.name+'-macpo');task=out/'macpo.task.json'
        write(task,dict(control=str(out),run=str(folder),result=str(out/'macpo.result.json'),config=config,scope='macpo-huber-'+args.scope,upstreams=str(upstreams),runtime=runtime,supervisor_pid=os.getpid(),supervisor_identity=state['identity'],evaluation_seeds=[54001,54002] if args.scope=='integration' else list(range(54001,54021)),variant='macpo-ot-symmetric-huber-v2',stream_training=True,reward_diagnostics=True))
        child=None;identity=None
        try:
            with (out/'macpo.log').open('x') as log:
                child=subprocess.Popen([sys.executable,str(out/'runner.py'),'--task',str(task)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(child.pid);save(child_pid=child.pid,child_identity=identity,current_run=str(folder));started=time.monotonic();sent=None
                while child.poll() is None:
                    if (out/'STOP').exists() or shutil.disk_usage(runs).free<30*1024**3 or time.monotonic()-started>21600:stopped=True
                    if stopped and sent is None and process_identity(child.pid)==identity:child.send_signal(signal.SIGINT);sent=time.monotonic()
                    if sent is not None and time.monotonic()-sent>120 and process_identity(child.pid)==identity:os.killpg(child.pid,signal.SIGTERM)
                    save();time.sleep(5)
                if stopped:raise InterruptedError('Stopped by request/resource boundary')
                if child.returncode:raise RuntimeError('MACPO task failed; preserve log')
            result=json.loads((out/'macpo.result.json').read_text());save(state='phase_complete',result=result,child_pid=None,child_identity=None)
        except BaseException as error:
            if child is not None and child.poll() is None and process_identity(child.pid)==identity:
                child.send_signal(signal.SIGINT)
                try:child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    if process_identity(child.pid)==identity:os.killpg(child.pid,signal.SIGTERM)
                    child.wait(timeout=30)
            save(state='stopped' if isinstance(error,(KeyboardInterrupt,InterruptedError)) else 'failed',reason=str(error));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--repair-control',type=Path,required=True)
    parser.add_argument('--basis-control',type=Path,required=True)
    parser.add_argument('--preflight-control',type=Path)
    parser.add_argument('--scope',choices=('integration','learning'),required=True)
    supervise(parser.parse_args())
