"""Synchronous on-policy episode sampling, with optional macOS-safe spawn workers.

Only frozen actor/critic weights go to workers. Only Blue-policy inputs, sampled
logp/value/reward/cost transitions return to the learner. Ground Truth stays in
research replay files, written to unique episode directories by each worker.
"""
from collections import deque
from contextlib import AbstractContextManager, redirect_stdout, redirect_stderr
import io
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import random
import signal
import time
import traceback
import numpy as np
from cage4_protocol.store import SCHEMA, atomic_json
from ..envs.adapter import Cage4Adapter
from ..recording.telemetry import frame, safe, red_telemetry
from ..safety.costs import measure, summarize


class EpisodeCollector:
    def __init__(self, config, budget_rate, expected_steps, policy, adapter=None):
        self.config=config;self.rate=budget_rate;self.expected_steps=expected_steps;self.policy=policy
        self.adapter=adapter or Cage4Adapter(config['seed'],config['steps'],'FiniteStateRedAgent',
                                 observation=config.get('observation'),include_graph=False)

    def collect(self, episode, seed, folder=None, training=True, progress=None):
        # Sampling RNG must not perturb learner minibatch/sequential-order RNG.
        # This makes one-worker and N-worker runs comparable with the same seeds.
        import torch
        python_state=random.getstate();numpy_state=np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                return self._collect(episode,seed,folder,training,progress)
        finally:
            random.setstate(python_state);np.random.set_state(numpy_state)

    def _collect(self, episode, seed, folder=None, training=True, progress=None):
        import torch
        c=self.config;env=self.adapter;policy=self.policy
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        started=time.perf_counter();timing=dict(inference=0.,environment=0.,cost=0.,replay=0.)
        inputs=env.reset(seed);policy.history=[];policy.begin_episode()
        scores={a:0. for a in inputs};costs=[];rewards_once=[]
        observation_stats={a:[] for a in inputs}
        decision_counts={a:dict(high_risk_host_action=0,stale_high_risk_host_action=0) for a in inputs}
        cumulative=0.;terminal=False
        def record(step,data):
            if folder is None or not c.get('record_replay',True):return
            p=Path(folder)/'frames'/f'episode_{episode:06d}'/f'step_{step:06d}.json'
            if p.exists():raise FileExistsError(p)
            atomic_json(p,dict(schema=SCHEMA,episode=episode,step=step,**data))
        if folder is not None and c.get('record_replay',True):record(0,frame(env,episode,0))
        for step in range(1,c['steps']+1):
            t=time.perf_counter()
            actions=policy.act(inputs,training=training,deterministic=c.get('deterministic_eval',True))
            timing['inference']+=time.perf_counter()-t
            labels={a:dict(index=i,label=inputs[a].labels[i],legal_mask=True) for a,i in actions.items()}
            for a,i in actions.items():
                for k,v in env.encoders[a].decision_diagnostics(inputs[a].labels[i]).items():decision_counts[a][k]+=v
            t=time.perf_counter();inputs,rewards,done,truncated=env.step(actions);timing['environment']+=time.perf_counter()-t
            t=time.perf_counter();cost=measure(env.base);costs.append(cost);timing['cost']+=time.perf_counter()-t
            cumulative+=cost['cost']
            context=dict(cumulative_cost=cumulative,recent_cost=cost['cost'],nominal_budget=self.rate*self.expected_steps,
                         remaining_budget=float(np.clip(1-cumulative/max(self.rate*self.expected_steps,1e-12),-1,1)),
                         information_scope='learner/researcher only; not actor input')
            for a,e in env.encoders.items():observation_stats[a].append(e.diagnostics.copy())
            for a in scores:scores[a]+=rewards[a]
            rewards_once.append(float(np.mean(list(rewards.values()))))
            terminal=not inputs or all(done.get(a,False) or truncated.get(a,False) for a in scores)
            if training:policy.observe(rewards,cost['cost'],terminal=terminal)
            t=time.perf_counter()
            if folder is not None and c.get('record_replay',True):
                if step==1 or terminal or step%c.get('snapshot_every',50)==0:data=frame(env,episode,step,labels)
                else:data=dict(blue=labels,blue_observations=safe(env.raw),red=red_telemetry(env.base))
                data.update(rewards=rewards,safety=cost,decision_context=context,seed=seed)
                record(step,data)
            timing['replay']+=time.perf_counter()-t
            if progress and (step%50==0 or terminal):progress(episode,step)
            if terminal:break
        if not terminal:raise RuntimeError('Rollout ended before real terminal')
        summary=summarize(costs)
        row=dict(episode_seed=seed,seed=c['seed'],steps=step,agents=scores,
                 observation_diagnostics={a:{**{k:float(np.mean([s[k] for s in values])) for k in values[0]},
                      **decision_counts[a],'stale_state_action_rate':decision_counts[a]['stale_high_risk_host_action']/max(1,decision_counts[a]['high_risk_host_action'])}
                      for a,values in observation_stats.items()},
                 decision_context=context,budget_usage=summary['mean_cost']/self.rate if self.rate else None,
                 team_return=sum(scores.values()),official_team_return=sum(rewards_once),
                 official_script_compat_return=sum(rewards_once[:-1]),safety=summary,
                 safety_budget_rate=self.rate,budget_excess=summary['mean_cost']-self.rate,
                 budget_violated=summary['mean_cost']>self.rate)
        timing['total']=time.perf_counter()-started
        return dict(episode=episode,row=row,history=policy.history if training else [],timing=timing,worker_pid=os.getpid())


def _worker(connection, config, upstreams, rate, expected_steps, parent_pid):
    # BLAS/Torch threads are limited *before* model construction in each child.
    signal.signal(signal.SIGINT,signal.SIG_IGN)
    os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['VECLIB_MAXIMUM_THREADS']='1'
    try:
        # Parent reports runtime compatibility once; avoid N identical Gym and
        # pygame notices. Exceptions still reach the parent with a full traceback.
        notices=io.StringIO()
        with redirect_stdout(notices),redirect_stderr(notices):
            import torch
            from .backend import AuthorBaseline
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
            collector=EpisodeCollector(config,rate,expected_steps,None)
            policy=AuthorBaseline(collector.adapter.reset(),config,upstreams,rate)
            collector.policy=policy
        connection.send(('ready',os.getpid()))
        def progress(ep,step):
            if os.getppid()!=parent_pid:raise InterruptedError('Learner exited')
            connection.send(('progress',ep,step,os.getpid()))
        while True:
            command=connection.recv()
            if command[0]=='stop':break
            _,weights,episode,seed,folder,training=command
            policy.load_rollout_state(weights)
            del weights,command
            result=collector.collect(episode,seed,folder,training,progress)
            connection.send(('result',result))
            policy.history=[];policy.pending=None
    except (EOFError,BrokenPipeError):pass
    except BaseException:
        try:connection.send(('error',traceback.format_exc()))
        except (EOFError,BrokenPipeError):pass
    finally:connection.close()


class RolloutPool(AbstractContextManager):
    def __init__(self, workers, config, upstreams, rate, expected_steps, timeout=180):
        if type(workers) is not int or not 1<=workers<=28:raise ValueError('workers must be 1..28')
        self.connections=[];self.processes=[];self.timeout=timeout;self.worker_pids=[]
        ctx=mp.get_context('spawn')
        limits=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')
        old_env={key:os.environ.get(key) for key in limits}
        for key in limits:os.environ[key]='1'
        try:
            for i in range(workers):
                parent,child=ctx.Pipe()
                p=ctx.Process(target=_worker,args=(child,config,upstreams,rate,expected_steps,os.getpid()),name=f'cage4-env-{i}')
                p.start();child.close();self.connections.append(parent);self.processes.append(p)
            for conn,p in zip(self.connections,self.processes):
                if not conn.poll(timeout):raise TimeoutError(f'Worker {p.pid} did not initialize')
                message=conn.recv()
                if message[0]!='ready':raise RuntimeError(str(message))
                self.worker_pids.append(message[1])
        except BaseException:
            self.close(force=True);raise
        finally:
            for key,value in old_env.items():
                if value is None:os.environ.pop(key,None)
                else:os.environ[key]=value

    def collect(self, weights, jobs, folder, training=True, progress=None):
        pending=deque(jobs);active={};results={};last_message=time.monotonic()
        def dispatch(conn):
            episode,seed=pending.popleft()
            conn.send(('collect',weights,episode,seed,str(folder) if folder is not None else None,training))
            active[conn]=episode
        for conn in self.connections:
            if pending:dispatch(conn)
        while active:
            ready=wait(list(active),timeout=1)
            if not ready and time.monotonic()-last_message>self.timeout:raise TimeoutError('Rollout workers stopped responding')
            for conn in ready:
                try:message=conn.recv()
                except EOFError as exc:raise RuntimeError(f'Worker exited during episode {active[conn]}') from exc
                last_message=time.monotonic()
                if message[0]=='error':raise RuntimeError(message[1])
                if message[0]=='progress':
                    if progress:progress(*message[1:])
                    continue
                if message[0]!='result':raise RuntimeError('Unexpected worker message')
                result=message[1]
                if result['episode']!=active[conn]:raise RuntimeError('Worker episode assignment mismatch')
                results[result['episode']]=result
                del active[conn]
                if pending:dispatch(conn)
        return [results[episode] for episode,_ in jobs]

    def close(self, force=False):
        for conn,p in zip(self.connections,self.processes):
            if p.is_alive() and not force:
                try:conn.send(('stop',))
                except (BrokenPipeError,EOFError,OSError):pass
        for p in self.processes:
            if force and p.is_alive():p.terminate()
            p.join(timeout=3)
            if p.is_alive():p.terminate();p.join(timeout=3)
        for conn in self.connections:conn.close()
        self.processes=[];self.connections=[]

    def __exit__(self, exc_type, exc, tb):self.close(force=exc_type is not None)
