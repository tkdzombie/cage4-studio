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
        prune_runs([root])
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
        prune_runs([self.path.parent])


def prune_runs(roots, keep=5):
    """Permanently retain newest runs; never delete a live/unverifiable writer.

    All files in a removed run (including checkpoints and replay) are deleted.
    Only direct, recognized run folders are managed; no symlink is followed.
    macOS/Linux advisory locks serialize competing cleanup operations.
    """
    import fcntl
    import math
    import shutil
    from contextlib import ExitStack
    if type(keep) is not int or keep < 1:
        raise ValueError('keep must be a positive integer')
    roots = sorted({Path(root).resolve() for root in roots}, key=str)
    removed = []
    with ExitStack() as stack:
        for root in roots:
            root.mkdir(parents=True, exist_ok=True)
            lock = stack.enter_context((root/'.retention.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX)
        candidates = []
        for root in roots:
            for folder in root.iterdir():
                if folder.is_symlink() or not folder.is_dir(): continue
                manifest = read_json(folder/'manifest.json', {})
                if not isinstance(manifest,dict): continue
                if manifest.get('schema') != SCHEMA and not (folder/'metrics.jsonl').is_file(): continue
                state = read_json(folder/'status.json', {})
                if not isinstance(state,dict): continue
                stamp = state.get('started_at')
                if type(stamp) not in (int, float) or not math.isfinite(stamp):
                    record = folder/('manifest.json' if manifest else 'metrics.jsonl')
                    stamp = record.stat().st_mtime
                candidates.append((stamp, folder))
        candidates.sort(key=lambda item:(item[0], str(item[1])), reverse=True)
        protected = {p for _,p in candidates[:keep]}
        for _, folder in candidates:
            if folder in protected: continue
            state = run_status(folder)
            # Older formats cannot verify PID reuse: conservatively keep a live PID.
            if state.get('state') == 'unverified': continue
            if state.get('state') in ('running', 'starting'):
                if state.get('identity') or process_identity(state.get('pid')): continue
            if not (folder/'status.json').exists(): continue  # writer may be initializing
            shutil.rmtree(folder)
            removed.append(folder)
    return removed
