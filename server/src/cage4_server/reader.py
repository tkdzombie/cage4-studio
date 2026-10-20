"""Read-only bounded access to v1 and legacy records; never unpickle checkpoints."""
from pathlib import Path
import json
from collections import deque
from cage4_protocol.store import SCHEMA,read_json,run_status
MAX_JSON = 16*1024*1024

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
        for p in root.iterdir():
            if p.is_symlink() or not p.is_dir(): continue
            manifest=read(root,p/'manifest.json',{})
            if manifest.get('schema')==SCHEMA or (p/'metrics.jsonl').is_file():
                result.append(p)
    return sorted(result,key=lambda p:p.name,reverse=True)

def records(folder,kind,limit=10000):
    files=sorted((folder/kind).glob('episode_*/step_*.json'))
    return [x for p in files[-limit:] if (x:=read(folder,p)) is not None]

def metrics(folder,limit=10000):
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
