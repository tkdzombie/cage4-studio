"""Compare pinned upstream and corrected MACPO losses on a private cached batch.

No simulator transitions. Only discarded critic copies take optimizer steps.
"""
import argparse
from copy import copy
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import time
from types import MethodType
from probe_macpo_value_fit import digest,fork_critic,write


def run(args):
    project=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(project/'protocol'),str(project/'cage4-baselines/src')]
    import numpy as np
    import torch
    from cage4_protocol.store import process_identity
    from cage4_baselines.envs.adapter import Cage4Adapter
    from cage4_baselines.paper.finite_horizon import FiniteHorizonBaseline
    from cage4_baselines.paper.backend import fingerprint
    from cage4_baselines.paper.provenance import runtime_identity
    basis=args.basis_control.resolve();cache=args.cache_control.resolve();out=args.output.resolve()
    assert json.loads((basis/'state.json').read_text())['state']=='phase_complete'
    assert json.loads((cache/'state.json').read_text())['state']=='phase_complete'
    previous=json.loads((cache/'result.json').read_text());assert previous['original_policy_weights_unchanged']
    result=next(r for r in json.loads((basis/'state.json').read_text())['results'] if r['algorithm']=='macpo')
    metadata=json.loads((Path(result['run'])/'manifest.json').read_text());checkpoint=Path(result['checkpoint'])
    assert digest(checkpoint)==previous['checkpoint_sha256']
    runtime=runtime_identity();assert {k:v for k,v in runtime.items() if k!='implementation_sha256'}=={k:v for k,v in metadata['runtime'].items() if k!='implementation_sha256'}
    with (out.parent/'.constrained.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(out.parent).free>30*1024**3
        out.mkdir(mode=0o700);started=time.monotonic()
        write(out/'plan.json',dict(schema='cage4.macpo-huber-cache-plan/1',environment_transitions=0,policy_updates=0,max_cloned_critic_steps=400,max_seconds=3600,min_free_gib=30,cache_sha256=digest(cache/'legal-blue-critic-batch.npz'),checkpoint_sha256=digest(checkpoint),script_sha256=digest(Path(__file__)),helper_sha256=digest(Path(__file__).with_name('probe_macpo_value_fit.py')),runtime=runtime,basis_runtime=metadata['runtime'],fit_seeds=[54001,54002],heldout_seeds=[54003,54004],only_comparison='Pinned asymmetric Huber vs symmetric Huber, original value clipping retained. Copies discarded.'))
        write(out/'state.json',dict(state='running',pid=os.getpid(),identity=process_identity(os.getpid())))
        try:
            torch.set_num_threads(2);torch.manual_seed(101);np.random.seed(101)
            spec=importlib.util.spec_from_file_location('basis_runner',basis/'runner.py');runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)
            payload=torch.load(checkpoint,map_location='cpu',weights_only=True);assert runner.finite_tree(payload)
            config=metadata['config'];env=Cage4Adapter(101,500,observation=config['observation'],include_graph=False)
            policy=FiniteHorizonBaseline(env.reset(),config,Path.home()/'cage4-upstreams',.05);policy.load(payload['policy'])
            from macpo.algorithms.r_mappo.r_macpo import R_MACTRPO_CPO
            def weights():return {a:{k:fingerprint(getattr(p,k)) for k in ('actor','critic','cost_critic')} for a,p in policy.policies.items()}
            before=weights();data=np.load(cache/'legal-blue-critic-batch.npz')
            observations=torch.from_numpy(data['central']);assert len(observations)==1996
            zero=torch.zeros((1996,policy.args.recurrent_N,policy.args.hidden_size));mask=torch.ones((1996,1))
            def score(model,target):
                model.eval()
                with torch.no_grad():pred=model(observations,zero,mask)[0].reshape(-1)
                return {name:float(((pred[section]-target[section])**2).mean()) for name,section in (('fit_mse',slice(0,998)),('heldout_mse',slice(998,None)))}
            rows=[];steps=0
            for index,agent in enumerate(policy.names):
                p=policy.policies[agent];tr=policy.trainers[agent]
                for kind,module_name,opt_name,pred_key in (('reward','critic','critic_optimizer','reward_prediction'),('cost','cost_critic','cost_optimizer','cost_prediction')):
                    target=torch.from_numpy(data['reward_target'][:,index] if kind=='reward' else data['cost_target'])
                    old=torch.from_numpy(data[pred_key][:,index,None]);error=target-old.reshape(-1)
                    tail=float((error < -tr.huber_delta).float().mean())
                    for corrected in (False,True):
                        model,opt=fork_critic(getattr(p,module_name),getattr(p,opt_name),payload['policy'][agent]['optimizers'][opt_name]);criterion=copy(tr)
                        if not corrected:criterion.cal_value_loss=MethodType(R_MACTRPO_CPO.cal_value_loss,criterion)
                        series=[dict(step=0,**score(model,target))]
                        for step in range(1,21):
                            if (out/'STOP').exists() or time.monotonic()-started>3600 or shutil.disk_usage(out).free<30*1024**3:raise InterruptedError('STOP/resource boundary')
                            model.train();opt.zero_grad();pred=model(observations[:998],zero[:998],mask[:998])[0]
                            loss=criterion.cal_value_loss(pred,old[:998],target[:998,None],mask[:998]);g=torch.autograd.grad(loss,pred,retain_graph=True)[0]
                            (loss*criterion.value_loss_coef).backward()
                            if criterion._use_max_grad_norm:torch.nn.utils.clip_grad_norm_(model.parameters(),criterion.max_grad_norm)
                            opt.step();steps+=1
                            assert runner.finite_tree(model.state_dict()) and runner.finite_tree(opt.state_dict())
                            if step in (1,5,10,20):series.append(dict(step=step,loss=float(loss.detach()),zero_gradient_fraction=float((g.abs()<1e-12).float().mean()),**score(model,target)))
                        rows.append(dict(agent=agent,kind=kind,corrected_huber=corrected,initial_negative_tail_fraction=tail,series=series))
                        write(out/'progress.json',dict(completed_arms=len(rows),total_arms=20,cloned_steps=steps,heartbeat=time.time()))
                        del model,opt,criterion
            assert steps==400 and weights()==before
            aggregate={kind:{'corrected' if corrected else 'upstream':float(np.mean([r['series'][-1]['heldout_mse'] for r in rows if r['kind']==kind and r['corrected_huber']==corrected])) for corrected in (False,True)} for kind in ('reward','cost')}
            write(out/'result.json',dict(schema='cage4.macpo-huber-cache/1',environment_transitions=0,policy_parameter_updates=0,cloned_optimizer_steps=400,original_policy_weights_unchanged=True,aggregate=aggregate,rows=rows,runtime=runtime,note='Cached development-data critic fitting only. No claim of improved RL performance or risk.'))
            write(out/'state.json',dict(state='phase_complete',environment_transitions=0,policy_parameter_updates=0,cloned_optimizer_steps=400))
        except BaseException as e:
            write(out/'state.json',dict(state='stopped' if isinstance(e,(KeyboardInterrupt,InterruptedError)) else 'failed',reason=str(e)));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--basis-control',type=Path,required=True)
    parser.add_argument('--cache-control',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    run(parser.parse_args())
