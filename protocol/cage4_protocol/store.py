"""v1: one writer per run; immutable atomic records; readers never execute data."""
from __future__ import annotations
import json
import os
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

SCHEMA = 'cage4.run/1'

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, allow_nan=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        # Persist the directory entry on supported filesystems.
        dfd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)

def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default

def process_identity(pid):
    """PID + start time + command; rejects reused PIDs, no shell execution."""
    try:
        p = subprocess.run(['ps', '-p', str(int(pid)), '-o', 'lstart=', '-o', 'command='],
                           capture_output=True, text=True, timeout=3, check=False)
        return p.stdout.strip() if p.returncode == 0 else None
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired):
        return None

def run_status(folder):
    s = read_json(Path(folder)/'status.json', {'state':'unknown'})
    if s.get('state') == 'running':
        if not s.get('identity'):
            return {**s, 'state':'unverified', 'reason':'process identity unavailable'}
        identity = process_identity(s.get('pid'))
        if identity is None:
            try: os.kill(int(s.get('pid')),0)
            except ProcessLookupError:
                return {**s,'state':'interrupted','reason':'writer process missing'}
            except (PermissionError,ValueError,TypeError): pass
            return {**s,'state':'unverified','reason':'cannot inspect process identity'}
        if identity != s['identity']:
            return {**s, 'state':'interrupted', 'reason':'writer process missing or PID reused'}
        s = {**s, 'heartbeat_age_seconds':max(0, time.time()-s.get('heartbeat',0))}
    return s

class RunWriter:
    def __init__(self, root, config, metadata):
        root = Path(root).resolve()
        root.mkdir(parents=True, exist_ok=True)
        self.path = root / (time.strftime('%Y%m%d_%H%M%S', time.gmtime())+'_'+uuid.uuid4().hex[:8])
        self.path.mkdir(mode=0o700)
        self.state = {'schema':SCHEMA, 'state':'running', 'pid':os.getpid(),
                      'identity':process_identity(os.getpid()), 'started_at':time.time()}
        atomic_json(self.path/'manifest.json', {'schema':SCHEMA, **metadata})
        atomic_json(self.path/'config.json', config)
        self.status()
    def status(self, **values):
        self.state.update(values, heartbeat=time.time())
        atomic_json(self.path/'status.json', self.state)
    def record(self, kind, episode, step, data):
        if kind not in ('metrics','frames'):
            raise ValueError('Unknown record kind')
        path = self.path/kind/f'episode_{episode:06d}'/f'step_{step:06d}.json'
        if path.exists():
            raise FileExistsError(path)
        atomic_json(path, {'schema':SCHEMA, 'episode':episode, 'step':step, **data})
    def finish(self, state='finished', **kw):
        self.status(state=state, finished_at=time.time(), **kw)
