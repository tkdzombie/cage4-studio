"""Fresh H-MARL master PPO with frozen author subpolicies; one training seed."""
import os
os.environ.setdefault('RAY_DEDUP_LOGS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse, json, time, signal, shutil, subprocess, hashlib, pickle
from pathlib import Path
import numpy as np
import torch
import ray
from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.policy.sample_batch import SampleBatch

class ScaledRewardPPO(PPOTorchPolicy):
    def postprocess_trajectory(self, sample_batch, other_agent_batches=None, episode=None):
        sample_batch[SampleBatch.REWARDS] = sample_batch[SampleBatch.REWARDS] * self.config.get('learner_reward_scale',1.0)
        return super().postprocess_trajectory(sample_batch, other_agent_batches, episode)

from ray.rllib.algorithms.callbacks import DefaultCallbacks


class Metrics(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        agent='blue_agent_0_master' if 'blue_agent_0_master' in episode.get_agents() else 'blue_agent_0'
        info=episode.last_info_for(agent) or {}
        for k in ['team_return','ticks','mask_mismatches']:
            if k in info: episode.custom_metrics[k]=info[k]

def mapper(agent_id, *args, **kwargs): return 'Agent'+agent_id[11]+('_master' if agent_id.endswith('_master') else '')

def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=8);p.add_argument('--batch',type=int,default=8192);p.add_argument('--iterations',type=int,default=60);p.add_argument('--seed',type=int,default=321);p.add_argument('--run',required=True);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--reward-scale',type=float,default=1.0);p.add_argument('--protocol',default='repaired');p.add_argument('--gae-lambda',type=float,default=.95);p.add_argument('--epochs',type=int,default=5);p.add_argument('--minibatch',type=int,default=1024);p.add_argument('--warm-start');p.add_argument('--warm-start-value-scale',type=float,default=1.0);p.add_argument('--level',choices=['master','flat'],default='master')
    a=p.parse_args(); torch.set_num_threads(1)
    if a.level=='flat':
        from primitive_env import PrimitiveEnv
        EnvClass=PrimitiveEnv;env_source='primitive_env.py'
    else:
        from hmarl_env import MasterEnv
        EnvClass=MasterEnv;env_source='hmarl_env.py'
    out=Path(a.run).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    (out/'source').mkdir()
    for filename in ['train.py',env_source]:
        shutil.copy2(Path(__file__).parent/filename,out/'source'/filename)
    provenance={'ray':ray.__version__,'torch':torch.__version__,'source_sha256':{f:hashlib.sha256((out/'source'/f).read_bytes()).hexdigest() for f in ['train.py',env_source]}}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2))
    ray.init(include_dashboard=False, num_cpus=a.workers+2, log_to_driver=False)
    env=EnvClass({'seed':a.seed,'protocol':a.protocol})
    suffix='_master' if a.level=='master' else ''
    specs={f'Agent{i}{suffix}':PolicySpec(ScaledRewardPPO,env.observation_space[f'blue_agent_{i}{suffix}'],env.action_space[f'blue_agent_{i}{suffix}'],{'learner_reward_scale':a.reward_scale}) for i in range(5)}
    config=(PPOConfig().framework('torch').environment(EnvClass,env_config={'seed':a.seed,'protocol':a.protocol},disable_env_checking=True)
        .experimental(_disable_preprocessor_api=True)
        .rollouts(num_rollout_workers=a.workers,rollout_fragment_length='auto',batch_mode='complete_episodes')
        .resources(num_gpus=0,num_cpus_per_worker=1)
        .training(model={'custom_model':'my_model','fcnet_hiddens':[256,256]},train_batch_size=a.batch,sgd_minibatch_size=a.minibatch,num_sgd_iter=a.epochs,
            lr=a.lr,gamma=.99,lambda_=a.gae_lambda,entropy_coeff=.001,vf_clip_param=10000.0,grad_clip=.5)
        .multi_agent(policies=specs,policy_mapping_fn=mapper,policies_to_train=list(specs))
        .reporting(metrics_num_episodes_for_smoothing=1).callbacks(Metrics).debugging(seed=a.seed,log_level='ERROR'))
    algo=config.build()
    if a.warm_start:
        weights={k:pickle.load(open(Path(a.warm_start)/'policies'/k/'policy_state.pkl','rb'))['weights'] for k in specs}
        if a.warm_start_value_scale!=1.0:
            for policy_weights in weights.values():
                for name in policy_weights:
                    if name.startswith('internal_model._value_branch.'):
                        policy_weights[name]=policy_weights[name]*a.warm_start_value_scale
        algo.set_weights(weights)
        algo.workers.sync_weights()
    algo.save(str(out/'initial')); stop=False
    def request_stop(*_):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGINT,request_stop);signal.signal(signal.SIGTERM,request_stop)
    best=-float('inf');history=[];start=time.time();reason='budget'
    for i in range(a.iterations):
        r=algo.train(); cm=r.get('custom_metrics',{});ret=cm.get('team_return_mean',r['episode_reward_mean']/5)
        learners=r.get('info',{}).get('learner',{})
        row={'iteration':i+1,'seconds':time.time()-start,'env_steps':r.get('timesteps_total'),'return':ret,'episode_count':r.get('episodes_this_iter'),'min_return':cm.get('team_return_min'),'max_return':cm.get('team_return_max'),'learner':{k:v.get('learner_stats',{}) for k,v in learners.items()}}
        with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(row,default=float)+'\n')
        print(json.dumps({k:v for k,v in row.items() if k!='learner'}),flush=True)
        history.append(ret)
        if not np.isfinite(ret) or any(not np.isfinite(v) for stats in row['learner'].values() for v in stats.values() if isinstance(v,(float,int))):
            reason='nonfinite_learner';stop=True
        # Checkpoint is concrete and usable before a diagnosis/stop.
        if ret>best: best=ret;algo.save(str(out/'best'))
        if (i+1)%5==0:algo.save(str(out/'latest'))
        # This is a diagnostic stop, NOT a claim of convergence. Evaluate checkpoints separately.
        if len(history)>=5 and all(history[j]<history[j-1]-15 for j in range(len(history)-4,len(history))):
            reason='persistent_degradation';stop=True
        if len(history)>=15:
            old=np.mean(history[-15:-10]);mid=np.mean(history[-10:-5]);new=np.mean(history[-5:]);tol=max(25,abs(old)*.025)
            if new<mid-tol and mid<old-tol:reason='persistent_degradation';stop=True
            elif abs(new-mid)<tol and abs(mid-old)<tol:reason='plateau_candidate';stop=True
        if stop:reason='requested' if reason=='budget' else reason;break
    algo.save(str(out/'final'))
    (out/'summary.json').write_text(json.dumps({'status':'stopped','reason':reason,'iterations':len(history),'best_training_return':best,'last_return':history[-1],'note':'training metrics only; convergence requires matched checkpoint evaluation'},indent=2))
    algo.stop();ray.shutdown()
if __name__=='__main__':main()
