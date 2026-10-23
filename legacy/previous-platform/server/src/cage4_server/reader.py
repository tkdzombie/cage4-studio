"""Read-only bounded access to v1 and legacy records; never unpickle checkpoints."""
from pathlib import Path
import json
import math
import time
from collections import deque
from cage4_protocol.store import SCHEMA,read_json,run_status
MAX_JSON = 16*1024*1024
OT_SCHEMA = 'cage4.ot-baseline-test/1'

def safe_file(root,path):
    root,path = Path(root).resolve(),Path(path)
    resolved = path.resolve()
    if resolved!=root and root not in resolved.parents:
        raise ValueError('Path escapes configured run root')
    if path.is_symlink():
        raise ValueError('Symlink records are not supported')
    return resolved

def read(root,path,default=None):
    try:
        p=safe_file(root,path)
        if p.stat().st_size > MAX_JSON: return default
        return read_json(p,default)
    except (OSError,ValueError): return default

def runs(roots):
    result=[]
    for root in roots:
        root=Path(root).resolve()
        if not root.is_dir(): continue
        # The OT campaign has one named grouping directory. Do not scan checkpoints
        # or recursively traverse arbitrary experiment assets.
        parents=[root]
        group=root/'ot-continuity-v1'
        if group.is_dir() and not group.is_symlink(): parents.append(group)
        for p in (p for parent in parents for p in parent.iterdir()):
            if p.is_symlink() or not p.is_dir(): continue
            manifest=read(root,p/'manifest.json',{})
            if manifest.get('schema') in (SCHEMA,OT_SCHEMA) or (p/'metrics.jsonl').is_file():
                result.append(p)
    def started(folder):
        state = read(folder, folder/'status.json', {})
        stamp = state.get('started_at') if isinstance(state, dict) else None
        if type(stamp) in (int, float) and math.isfinite(stamp):
            return (stamp, folder.name)
        # Legacy records have no common timestamp: use record metadata, not a
        # directory prefix or directory mtime affected by later housekeeping.
        for name in ('manifest.json', 'metrics.jsonl', 'status.json'):
            try:
                return (safe_file(folder, folder/name).stat().st_mtime, folder.name)
            except (OSError, ValueError):
                continue
        return (0, folder.name)
    return sorted(set(result), key=started, reverse=True)

def records(folder,kind,limit=10000):
    files=sorted((folder/kind).glob('episode_*/step_*.json'))
    return [x for p in files[-limit:] if (x:=read(folder,p)) is not None]

def metrics(folder,limit=10000):
    metadata=read(folder,folder/'manifest.json',{})
    if metadata.get('schema')==OT_SCHEMA:
        # Live JSONL is authoritative; the JSON snapshot can lag by a batch.
        rows=[]
        if (folder/'training.jsonl').exists():
            try:
                with safe_file(folder,folder/'training.jsonl').open(encoding='utf-8') as f:
                    tail=deque(enumerate(f,1),maxlen=limit)
                for episode,line in tail:
                    try: rows.append((episode,json.loads(line)))
                    except ValueError: continue  # incomplete current append
            except (OSError,ValueError): pass
        else:
            rows=list(enumerate(read(folder,folder/'training.json',[]),1))[-limit:]
        updates={u['episode']:u.get('metrics',{}) for u in read(folder,folder/'updates.json',[])}
        return [dict(episode=i,step=i*x['steps'],official_team_return=x['return_'],
                     ot_event=x['event_cost'],event_budget=metadata['event_budget'],
                     losses=updates.get(i,{})) for i,x in rows]
    if read(folder,folder/'manifest.json',{}).get('schema')==SCHEMA:
        return records(folder,'metrics',limit)
    try:
        path=safe_file(folder,folder/'metrics.jsonl')
        # Streaming tail handles incomplete legacy appends without loading the whole file.
        with path.open(encoding='utf-8') as f: lines=deque(f,maxlen=limit)
        rows=[]
        for line in lines:
            try: rows.append(json.loads(line))
            except ValueError: continue
        return rows
    except (OSError,ValueError): return []

def status(folder):
    if read(folder,folder/'manifest.json',{}).get('schema')==OT_SCHEMA:
        try: safe_file(folder,folder/'status.json')
        except ValueError: return {'state':'invalid status path'}
        state=run_status(folder)
        progress=read(folder,folder/'progress.json',{})
        rows=metrics(folder)
        completed=max(progress.get('completed_training',0),rows[-1]['episode'] if rows else 0)
        step=completed*499+(progress.get('step',0) if progress.get('training') else 0)
        if progress.get('completed_training',0)<completed: step=completed*499
        return {**state,'completed_episodes':completed,'step':step,
                'phase':'collecting' if progress.get('training',True) else 'evaluating',
                'heartbeat_age_seconds':max(0,time.time()-progress.get('heartbeat',0))}
    if read(folder,folder/'manifest.json',{}).get('schema')==SCHEMA:
        # Validate the path first, then shared process identity logic.
        try: safe_file(folder,folder/'status.json')
        except ValueError: return {'state':'invalid status path'}
        return run_status(folder)
    data=read(folder,folder/'status.json',{})
    if data.get('state') in ('running','starting'):
        data={**data,'state':'legacy_unverified','reason':'legacy PID lacks process identity'}
    return data

def frame_paths(folder):
    if read(folder,folder/'manifest.json',{}).get('schema')==SCHEMA:
        return sorted((folder/'frames').glob('episode_*/step_*.json'))
    return sorted((folder/'snapshots').glob('episode_*/step_*.json'))
