from pathlib import Path
import math
import yaml
DEFAULTS = dict(algorithm='actor_critic',seed=7629,steps=100,episodes=30,
                red='FiniteStateRedAgent',lr=0.0003,gamma=0.99,entropy_coef=0.01,
                device='cpu',snapshot_every=10,checkpoint_every=5,
                tensorboard=False,plugins=[],torch_threads=4)
def load(path):
    raw = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(raw,dict) or set(raw)-set(DEFAULTS):
        raise ValueError('Config must be a mapping of documented keys')
    c = {**DEFAULTS,**raw}
    for key,lo,hi in [('seed',0,2**32-1),('steps',10,1000),('episodes',1,100000),
                      ('snapshot_every',1,1000),('checkpoint_every',1,100000),('torch_threads',1,8)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi:
            raise ValueError('Invalid '+key)
    for key,lo,hi in [('lr',1e-9,.1),('gamma',1e-9,1.),('entropy_coef',0.,1.)]:
        if not isinstance(c[key],(float,int)) or not math.isfinite(c[key]) or not lo<=c[key]<=hi:
            raise ValueError('Invalid '+key)
    if c['device'] not in ('cpu','mps') or c['red'] not in ('FiniteStateRedAgent','DiscoveryFSRed'):
        raise ValueError('Unsupported device/red')
    if not isinstance(c['algorithm'],str) or type(c['tensorboard']) is not bool:
        raise ValueError('Invalid algorithm/tensorboard')
    if not isinstance(c['plugins'],list) or not all(isinstance(v,str) for v in c['plugins']):
        raise ValueError('plugins must be module names')
    return c
