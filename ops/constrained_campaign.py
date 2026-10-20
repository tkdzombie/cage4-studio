"""Bounded, sequential constrained-baseline experiments with frozen source.

The server runs this supervisor independently of the desktop. It never tunes
against the final test set or declares paper reproduction from a pilot.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ALGORITHMS = ('macpo', 'mappo_lagrangian')
DEVELOPMENT_SEEDS = list(range(54001, 54021))


def build_plan(algorithms, seeds, episodes):
    if not algorithms or len(set(algorithms)) != len(algorithms) or any(a not in ALGORITHMS for a in algorithms):
        raise ValueError('Only unique constrained algorithms are allowed')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 1 <= s <= 1000 for s in seeds):
        raise ValueError('Invalid training seeds')
    if type(episodes) is not int or not 1 <= episodes <= 50000:
        raise ValueError('Invalid episode count')
    return [dict(algorithm=a, seed=s, episodes=episodes) for s in seeds for a in algorithms]



def load_overrides(path, algorithms):
    """Only explicit, single-hyperparameter candidates; never protocol changes."""
    if path is None:
        return {}
    data = json.loads(Path(path).read_text())
    if (not isinstance(data, dict) or set(data) != {'schema', 'algorithms'}
            or data['schema'] != 'cage4.campaign-overrides/1'):
        raise ValueError('Invalid campaign override schema')
    values = data['algorithms']
    if not isinstance(values, dict) or not values or set(values)-set(algorithms):
        raise ValueError('Overrides must select algorithms in this campaign')
    allowed = dict(macpo={'fraction_coef', 'kl_threshold', 'ls_step', 'critic_lr'},
                   mappo_lagrangian={'lr', 'critic_lr', 'entropy_coef', 'lamda_lagr', 'lagrangian_coef_rate'})
    for name, candidate in values.items():
        if not isinstance(candidate, dict) or len(candidate) != 1 or set(candidate)-allowed[name]:
            raise ValueError('One supported hyperparameter per algorithm is required')
        key, value = next(iter(candidate.items()))
        if key == 'ls_step':
            valid = type(value) is int and 1 <= value <= 100
        else:
            low = 0 if key in ('entropy_coef', 'lamda_lagr') else 1e-10
            high = .1 if key in ('lr', 'critic_lr', 'lagrangian_coef_rate') else (100 if key == 'lamda_lagr' else 1)
            valid = type(value) in (int, float) and math.isfinite(value) and low <= value <= high
        if not valid:
            raise ValueError('Invalid candidate value: '+key)
    return values


def training_summary(folder):
    rows = [json.loads(p.read_text()) for p in sorted((Path(folder)/'metrics').glob('episode_*/step_*.json'))]
    result = dict(completed_episodes=len(rows))
    for label, values in (('first100', rows[:100]), ('last100', rows[-100:])):
        if values:
            result[label] = dict(n=len(values), mean_return=sum(r['official_team_return'] for r in values)/len(values),
                                 mean_cost=sum(r['safety']['mean_cost'] for r in values)/len(values),
                                 violation_fraction=sum(r['budget_violated'] for r in values)/len(values))
    updates = [r for r in rows if r.get('optimizer_update')]
    result['optimizer_updates'] = len(updates)
    result['last_losses'] = updates[-1]['losses'] if updates else {}
    return result


def task_main(path):
    task = json.loads(Path(path).read_text())
    # Only this locally created, trusted task file can select frozen source.
    sys.path.insert(0, task['source'])
    from cage4_baselines.paper.__main__ import run
    from cage4_protocol.store import atomic_json
    options = dict(mode=task['mode'], smoke=task['smoke'])
    if task['mode'] == 'evaluate':
        options.update(checkpoint=Path(task['checkpoint']),seeds=task['evaluation_seeds'],
                       evaluation_action_mode=task['action_mode'])
    folder = run(task['config'], Path(task['runs']), Path(task['upstreams']), Path(task['budget']), **options)
    atomic_json(Path(task['result']),dict(run=str(folder)))


@contextmanager
def campaign_lock(parent):
    Path(parent).mkdir(parents=True, exist_ok=True)
    with (Path(parent)/'.constrained.lock').open('a') as handle:
        try: fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise RuntimeError('Another constrained supervisor owns this control root')
        yield


def supervise(a):
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(project/'protocol'),str(project/'cage4-baselines/src')]
    from cage4_protocol.store import atomic_json, read_json, process_identity, run_status
    from cage4_baselines.paper.__main__ import load_config
    from cage4_baselines.paper.provenance import runtime_identity
    from cage4_baselines.safety.budgets import read_budget
    plan = build_plan(a.algorithms,a.seeds,a.episodes)
    overrides = load_overrides(a.overrides, a.algorithms)
    control = a.control.expanduser().resolve()
    runs = a.runs.expanduser().resolve(); runs.mkdir(parents=True,exist_ok=True)
    stopping = False
    def interrupt(signum,frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGINT,interrupt);signal.signal(signal.SIGTERM,interrupt)
    with campaign_lock(control.parent):
        for folder in runs.iterdir():
            if folder.is_dir() and run_status(folder).get('state') in ('running','unverified'):
                raise RuntimeError('Existing training must finish or be stopped before starting a campaign')
        control.mkdir(mode=0o700)  # Never overwrite or silently restart a campaign.
        source = control/'source'; source.mkdir()
        shutil.copytree(project/'cage4-baselines/src/cage4_baselines',source/'cage4_baselines',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copytree(project/'protocol/cage4_protocol',source/'cage4_protocol',
                        ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        shutil.copy2(__file__,control/'worker.py')
        shutil.copytree(project/'cage4-baselines/configs/paper',control/'configs')
        shutil.copy2(project/'cage4-baselines/configs/safety/availability-target-v2.json',control/'budget.json')
        _,rate = read_budget(control/'budget.json',500,level=a.budget_level)
        atomic_json(control/'overrides.json',dict(schema='cage4.campaign-overrides/1',algorithms=overrides))
        state = dict(schema='cage4.constrained-campaign/1',state='running',pid=os.getpid(),
                     identity=process_identity(os.getpid()),started_at=time.time(),runtime=runtime_identity(),
                     primary_action_mode='stochastic',secondary_action_mode='greedy',budget_rate=rate,
                     budget_level=a.budget_level,configuration_overrides=overrides,
                     development_seeds=DEVELOPMENT_SEEDS,smoke=a.smoke,plan=plan,results=[])
        def save(**values):
            state.update(values,heartbeat=time.time());atomic_json(control/'state.json',state)
        save()

        def execute(task, label):
            nonlocal stopping
            task_path=control/(label+'.task.json');result_path=control/(label+'.result.json')
            task.update(source=str(source),upstreams=str(a.upstreams.expanduser().resolve()),
                        budget=str(control/'budget.json'),result=str(result_path),smoke=a.smoke)
            atomic_json(task_path,task)
            with (control/(label+'.log')).open('w') as log:
                child=subprocess.Popen([sys.executable,str(control/'worker.py'),'--task',str(task_path)],
                                       stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                identity=process_identity(child.pid)
                save(phase=task['mode'],current_task=label,child_pid=child.pid,child_identity=identity,current_run=None)
                started=time.monotonic();sent=None;run_folder=None
                try:
                    while child.poll() is None:
                        if run_folder is None and task['mode']=='train':
                            for folder in runs.iterdir():
                                status=read_json(folder/'status.json',{}) if folder.is_dir() else {}
                                if status.get('pid')==child.pid and status.get('identity')==identity:
                                    run_folder=folder;break
                        low_disk=shutil.disk_usage(runs).free < a.reserve_gib*1024**3
                        if low_disk or time.monotonic()-started>a.max_job_hours*3600:
                            stopping=True
                            save(reason='disk reserve reached' if low_disk else 'job wall-time limit reached')
                        if (control/'STOP').exists():stopping=True
                        if stopping and sent is None:
                            if identity is not None and process_identity(child.pid)==identity:
                                child.send_signal(signal.SIGINT);sent=time.monotonic()
                        elif sent is not None and time.monotonic()-sent>120:
                            if process_identity(child.pid)==identity:
                                os.killpg(child.pid,signal.SIGTERM)
                        current=run_status(run_folder) if run_folder else None
                        save(current_run=str(run_folder) if run_folder else None,run_status=current,
                             free_gib=shutil.disk_usage(runs).free/1024**3)
                        time.sleep(5)
                finally:
                    if child.poll() is None:
                        if identity is not None and process_identity(child.pid)==identity:
                            child.send_signal(signal.SIGINT)
                        try:child.wait(timeout=120)
                        except subprocess.TimeoutExpired:
                            if process_identity(child.pid)==identity:os.killpg(child.pid,signal.SIGTERM)
                            child.wait(timeout=30)
                if stopping:raise InterruptedError('Campaign stopped')
                if child.returncode:
                    raise RuntimeError(f'{label} exited {child.returncode}; inspect {label}.log')
            folder=Path(read_json(result_path)['run'])
            return folder

        try:
            for index, item in enumerate(plan,1):
                if stopping or (control/'STOP').exists():raise InterruptedError('Campaign stopped')
                c=load_config(control/'configs'/(item['algorithm']+'.yaml'))
                c.update(episodes=item['episodes'],seed=item['seed'],deterministic_eval=False,budget_level=a.budget_level)
                c['hyperparameters']=dict(c.get('hyperparameters',{}),**overrides.get(item['algorithm'],{}))
                if a.smoke:
                    c.update(steps=20,num_envs=2,torch_threads=1,record_replay=False)
                    c['hyperparameters']=dict(c['hyperparameters'],hidden_size=32,ppo_epoch=2)
                label=f'{index:02d}-{item["algorithm"]}-{item["seed"]}'
                save(job_index=index,current_algorithm=item['algorithm'])
                try:
                    folder=execute(dict(mode='train',config=c,runs=str(runs)),label+'-train')
                    result=dict(**item,run=str(folder),training=training_summary(folder),evaluations=[])
                    state['results'].append(result);save()
                    checkpoints=sorted({min(200,item['episodes']),min(1000,item['episodes']),item['episodes']})
                    evaluation_seeds=DEVELOPMENT_SEEDS[:3] if a.smoke else DEVELOPMENT_SEEDS
                    for episode in checkpoints:
                        checkpoint=folder/'checkpoints'/f'episode_{episode:06d}.pt'
                        if not checkpoint.is_file():continue
                        for action_mode in ('stochastic','greedy'):
                            evaluated=execute(dict(mode='evaluate',config=c,runs=str(control/'evaluation-work'),
                                checkpoint=str(checkpoint),evaluation_seeds=evaluation_seeds,action_mode=action_mode),
                                label+f'-eval-{episode}-{action_mode}')
                            report=read_json(evaluated/'evaluation.json')
                            result['evaluations'].append(dict(checkpoint_episode=episode,**report))
                            save()
                            shutil.rmtree(evaluated)  # Metrics are preserved above; replay is temporary.
                    result['state']='finished';save()
                except InterruptedError:raise
                except Exception as error:
                    state.setdefault('errors',[]).append(dict(job=item,error=str(error)));save()
            save(state='needs_review' if state.get('errors') else 'phase_complete',phase='review',finished_at=time.time())
        except InterruptedError:
            save(state='stopped',phase='stopped',finished_at=time.time())
        except BaseException as error:
            save(state='error',error=str(error),finished_at=time.time());raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--control',type=Path)
    p.add_argument('--runs',type=Path,default=Path.home()/'cage4-experiments')
    p.add_argument('--upstreams',type=Path,default=Path.home()/'cage4-upstreams')
    p.add_argument('--algorithms',nargs='+',choices=ALGORITHMS,default=list(ALGORITHMS))
    p.add_argument('--seeds',nargs='+',type=int,default=[101])
    p.add_argument('--episodes',type=int,default=2000)
    p.add_argument('--overrides',type=Path,help='JSON single-hyperparameter candidates, frozen per campaign')
    p.add_argument('--budget-level',choices=('main','strict','loose'),default='main')
    p.add_argument('--reserve-gib',type=float,default=30.)
    p.add_argument('--max-job-hours',type=float,default=6.)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.task:return task_main(a.task)
    if a.control is None:p.error('--control is required')
    if not math.isfinite(a.reserve_gib) or not math.isfinite(a.max_job_hours) or a.reserve_gib<1 or a.max_job_hours<=0:
        p.error('Invalid resource limits')
    supervise(a)


if __name__=='__main__':main()
