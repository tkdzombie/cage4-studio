"""Small, deterministic, observation-only belief and degradation primitives."""
from copy import deepcopy
from collections import deque
import math
import numpy as np

CHANNELS = ('process', 'connection', 'file', 'host')
DEFAULTS = dict(mode='belief', age_max=500, decay=.01, dropout=0., delay=0, noise=0., seed=0)


def observation_config(config=None):
    c = {**DEFAULTS, **(config or {})}
    if set(c)-set(DEFAULTS): raise ValueError('Unknown observation option')
    if c['mode'] not in ('current','persistent','age','belief'): raise ValueError('Invalid observation mode')
    for key,lo,hi in [('age_max',1,10000),('delay',0,100),('seed',0,2**32-1)]:
        if type(c[key]) is not int or not lo<=c[key]<=hi: raise ValueError('Invalid '+key)
    for key,lo,hi in [('decay',0.,1.),('dropout',0.,1.),('noise',0.,1.)]:
        if type(c[key]) not in (int,float) or not math.isfinite(c[key]) or not lo<=c[key]<=hi:
            raise ValueError('Invalid '+key)
    return c


def update_event_belief(last_observed, channel, source_step):
    last_observed[channel] = source_step


def update_confidence(last_observed, step, age_max, decay):
    """Recency weights, not calibrated probabilities of truth."""
    ages = [age_max if last_observed[k] is None else max(0,step-last_observed[k]) for k in CHANNELS]
    observed = [last_observed[k] == step for k in CHANNELS]
    linear = [min(a,age_max)/age_max for a in ages]
    logarithmic = [math.log1p(min(a,age_max))/math.log1p(age_max) for a in ages]
    confidence = [0. if last_observed[k] is None else math.exp(-decay*a) for k,a in zip(CHANNELS,ages)]
    return observed,linear,logarithmic,confidence


class ObservationTransport:
    """Corrupt deep copies only; separate RNG per agent, reseeded per episode."""
    def __init__(self, config, hosts):
        self.config=observation_config(config);self.hosts=set(hosts);self.queue=deque()
        self.rng=np.random.default_rng(self.config['seed'])

    def push(self, observation, step):
        packet=deepcopy(observation)
        dropped=set()
        for host in sorted(self.hosts & packet.keys()):
            if self.rng.random()<self.config['dropout']:
                del packet[host];dropped.add(host)
        # A successful Analyse may return an empty host payload. Drop its query
        # result too, rather than incorrectly turning packet loss into a negative.
        action=packet.get('action');target=getattr(action,'hostname',None)
        if isinstance(action,str):
            parts=action.split();target=parts[1] if len(parts)>1 else None
        if target in self.hosts and target not in dropped and target not in packet:
            if self.rng.random()<self.config['dropout']:dropped.add(target)
        self.queue.append((packet,step,dropped))
        if len(self.queue)<=self.config['delay']: return {},None,set(self.hosts)
        return self.queue.popleft()

    def noisy(self, value, observed):
        if observed and self.rng.random()<self.config['noise']: return not value
        return value
