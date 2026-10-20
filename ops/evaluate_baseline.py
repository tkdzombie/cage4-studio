"""Evaluate a trusted saved run using its exact JSON config and checkpoint.

JSON scientific-notation numbers must be read as JSON, not YAML 1.1 strings.
This command lives outside policy runtime sources so input/CLI repairs do not
invalidate an already running experiment's algorithm provenance.
"""
import argparse
import json
import sys
from pathlib import Path


def read_training_config(folder):
    config=json.loads((Path(folder)/'config.json').read_text(encoding='utf-8'))
    if not isinstance(config,dict) or not {'algorithm','seed','steps','episodes','hyperparameters'}<=config.keys():
        raise ValueError('Not a recorded baseline configuration')
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--episode',type=int,required=True)
    p.add_argument('--seeds',type=int,nargs='+',required=True)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--upstreams',type=Path,default=Path.home()/'cage4-upstreams')
    p.add_argument('--budget',type=Path)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--action-mode',choices=('stochastic','greedy'))
    a=p.parse_args()
    snapshot=a.run.resolve()/'source'
    if (snapshot/'cage4_baselines'/'paper'/'__main__.py').is_file():
        sys.path.insert(0,str(snapshot))
    from cage4_baselines.paper.__main__ import run
    config=read_training_config(a.run)
    checkpoint=a.run/'checkpoints'/f'episode_{a.episode:06d}.pt'
    budget=a.budget or a.run/'budget.json'
    options={} if a.action_mode is None else dict(evaluation_action_mode=a.action_mode)
    print(run(config,a.runs,a.upstreams,budget,mode='evaluate',checkpoint=checkpoint,
              seeds=a.seeds,smoke=a.smoke,**options),flush=True)

if __name__=='__main__':main()
