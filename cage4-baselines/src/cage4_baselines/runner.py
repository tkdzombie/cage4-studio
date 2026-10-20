from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import platform
import importlib.metadata
import random
import signal
import tempfile
import traceback
import numpy as np
from cage4_protocol.store import RunWriter, atomic_json
from .adapter import Cage4Adapter
from .algorithms import create
from .config import load
from .observations import ENCODER
from .telemetry import frame, safe, red_telemetry

def save_checkpoint(path, payload):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd,temp = tempfile.mkstemp(prefix='.',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as f:
            torch.save(payload,f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp,path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def run(config, runs_root, mode='train', checkpoint=None, seeds=None, adapter_factory=Cage4Adapter):
    c = dict(config)
    if mode=='evaluate' and c['algorithm']=='actor_critic' and checkpoint is None:
        raise ValueError('Actor-Critic evaluation requires --checkpoint')
    if mode=='train' and checkpoint:
        raise ValueError('Resume training is not implemented; checkpoint is evaluation only')
    seeds = [c['seed']] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(s<0 or s>=2**32 for s in seeds):
        raise ValueError('Evaluation seeds must be unique uint32 integers')
    versions = {}
    for name in ('torch','numpy','gymnasium','ray','pyarrow','setuptools','PyYAML','CybORG'):
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    metadata = {'dependencies':versions,'mode':mode,'algorithm':c['algorithm'],'encoder':ENCODER,
                'python':platform.python_version(),'platform':platform.platform(),
                'evaluation_seeds':seeds if mode=='evaluate' else [],
                'reward_semantics':'sum of five official wrapper returns; shared team rewards may be repeated',
                'checkpoint_sha256':hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() if checkpoint else None}
    writer = RunWriter(runs_root,c,metadata)
    stopped = [False]
    old_handlers = {}
    def stop(*_): stopped[0] = True
    for sig in (signal.SIGINT,signal.SIGTERM):
        old_handlers[sig] = signal.signal(sig,stop)
    tb = None
    try:
        if c['tensorboard']:
            from torch.utils.tensorboard import SummaryWriter
            tb = SummaryWriter(str(writer.path/'tensorboard'))
        random.seed(c['seed']); np.random.seed(c['seed'])
        adapter = adapter_factory(c['seed'],c['steps'],c['red'],c['obs_dim'])
        inputs = adapter.inputs(adapter.reset(c['seed']))
        policy = create(c['algorithm'],inputs,c)
        if checkpoint:
            import torch
            # Only load trusted local checkpoints. Viewer never deserializes .pt files.
            payload = torch.load(checkpoint,map_location=c['device'],weights_only=True)
            if payload['encoder'] != ENCODER or payload['config']['algorithm'] != c['algorithm']:
                raise ValueError('Incompatible checkpoint')
            policy.load(payload['policy'])
        atomic_json(writer.path/'spaces.json',{a:{'observation_dim':len(x.observation),
                     'actions':len(x.mask)} for a,x in inputs.items()})
        rows=[]
        episode = 0
        for seed in seeds:
            if mode=='evaluate':
                random.seed(seed); np.random.seed(seed)
                if hasattr(policy,'rng'): policy.rng = np.random.default_rng(seed)
                if hasattr(policy,'torch'): policy.torch.manual_seed(seed)
            for local_ep in range(c['episodes']):
                if stopped[0]: break
                episode += 1
                # Distinct deterministic episode seeds; evaluation seeds are explicitly recorded.
                episode_seed = seed if local_ep==0 else int(np.random.SeedSequence([seed,local_ep]).generate_state(1)[0])
                inputs = adapter.inputs(adapter.reset(episode_seed))
                writer.record('frames',episode,0,frame(adapter,episode,0))
                scores={a:0. for a in inputs}
                last = 0
                for step in range(1,c['steps']+1):
                    if stopped[0]: break
                    acts = policy.act(inputs,training=mode=='train')
                    labels = {a:{'index':i,'label':inputs[a].labels[i],
                                 'legal_mask':bool(inputs[a].mask[i])} for a,i in acts.items()}
                    next_inputs,rewards,terminated,truncated = adapter.step(acts)
                    if mode=='train': policy.observe(rewards)
                    for a in scores: scores[a]+=rewards.get(a,0.)
                    terminal = not next_inputs or all(terminated.get(a,False) or truncated.get(a,False) for a in scores)
                    # One atomic frame for every executed step: observations/actions/rewards replay.
                    # Full ground-truth topology is sparse to control storage.
                    if step % c['snapshot_every']==0 or step==1 or step==c['steps'] or terminal:
                        data = frame(adapter,episode,step,labels)
                    else:
                        data = {'blue':labels,'blue_observations':safe(adapter.raw),'red':red_telemetry(adapter.base)}
                    data.update(rewards=rewards,terminated=terminated,truncated=truncated,seed=episode_seed)
                    writer.record('frames',episode,step,data)
                    writer.status(episode=episode,step=step,seed=episode_seed)
                    inputs = adapter.inputs(next_inputs)
                    last = step
                    if terminal: break
                if stopped[0]: break
                losses = policy.update() if mode=='train' else {}
                row = {'seed':seed,'episode_seed':episode_seed,'seed_episode':local_ep+1,'steps':last,
                       'team_return':sum(scores.values()),'agents':scores,'losses':losses}
                writer.record('metrics',episode,last,row)
                rows.append(row)
                if tb:
                    tb.add_scalar('team_return',row['team_return'],episode)
                    for a,score in scores.items():
                        tb.add_scalar('return/'+a,score,episode)
                    for a,values in losses.items():
                        for key,value in values.items(): tb.add_scalar(key+'/'+a,value,episode)
                    tb.flush()
                if mode=='train' and (episode % c['checkpoint_every']==0 or local_ep==c['episodes']-1):
                    save_checkpoint(writer.path/'checkpoints'/f'episode_{episode:06d}.pt',
                                    {'encoder':ENCODER,'config':c,'episode':episode,'policy':policy.checkpoint()})
        if mode=='evaluate':
            groups = {str(seed):[x['team_return'] for x in rows if x['seed']==seed] for seed in seeds}
            means = [float(np.mean(x)) for x in groups.values() if x]
            atomic_json(writer.path/'evaluation.json',{'by_seed':groups,'seed_means':means,
                        'mean':float(np.mean(means)) if means else None,
                        'sample_std':float(np.std(means,ddof=1)) if len(means)>1 else None,
                        'n_seeds':len(means),'complete':not stopped[0]})
        writer.finish('stopped' if stopped[0] else 'finished')
        return writer.path
    except BaseException as exc:
        writer.finish('error',error=str(exc),traceback=traceback.format_exc())
        raise
    finally:
        if tb: tb.close()
        for sig,handler in old_handlers.items(): signal.signal(sig,handler)

def main():
    p=argparse.ArgumentParser(description='Independent CAGE4 experiment runner')
    p.add_argument('mode',choices=['train','evaluate'])
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--runs',required=True,type=Path)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--seeds',type=int,nargs='+')
    a=p.parse_args()
    if a.mode=='train' and a.seeds:
        p.error('--seeds is for evaluation; use independent train configurations per seed')
    print(run(load(a.config),a.runs,a.mode,a.checkpoint,a.seeds),flush=True)
if __name__=='__main__': main()
