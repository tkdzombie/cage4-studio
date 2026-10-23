"""Measure synchronous MAPPO throughput; outputs live outside research run roots.

No shortened horizon or reduced model is used unless --smoke is explicit.
Worker counts share the exact gradient batch size and episode RNG protocol.
"""
import argparse
import json
from pathlib import Path
import time
from cage4_baselines.paper.__main__ import load_config,run
from cage4_protocol.store import atomic_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--upstreams',type=Path,required=True)
    p.add_argument('--workers',type=int,nargs='+',default=[1,4,8,16])
    p.add_argument('--threads',type=int,nargs='+',default=[1,4])
    p.add_argument('--episodes',type=int,default=16)
    p.add_argument('--batch',type=int,default=16)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--replay',action='store_true')
    a=p.parse_args();project=Path(__file__).resolve().parents[1]
    results=[]
    for threads in a.threads:
      for workers in a.workers:
        c=load_config(project/'cage4-baselines/configs/paper/mappo.yaml')
        c.update(episodes=a.episodes,rollout_episodes=a.batch,num_envs=workers,torch_threads=threads,
                 checkpoint_every=50,record_replay=a.replay)
        if a.smoke:c.update(steps=20,hyperparameters=dict(hidden_size=32,layer_N=1,ppo_epoch=2))
        start=time.perf_counter()
        folder=run(c,a.output/f'w{workers}-t{threads}',a.upstreams,project/'cage4-baselines/configs/safety/availability-v1.json',smoke=a.smoke)
        elapsed=time.perf_counter()-start
        perf=json.loads((folder/'performance.json').read_text())
        metrics=[json.loads(f.read_text()) for f in sorted((folder/'metrics').rglob('*.json'))]
        results.append(dict(workers=workers,threads=threads,wall_seconds=elapsed,
                            transitions=perf['total_transitions'],end_to_end_fps=perf['total_transitions']/elapsed,
                            performance=perf,returns=[r['official_team_return'] for r in metrics],
                            updates=[r['episode'] for r in metrics if r['optimizer_update']],
                            final_losses=metrics[-1]['losses']))
        atomic_json(a.output/'benchmark.json',dict(scope='throughput/correctness; not convergence',results=results))
        print('BENCHMARK_RESULT',json.dumps({k:v for k,v in results[-1].items() if k not in ('returns','final_losses','performance')}),flush=True)

if __name__=='__main__':main()
