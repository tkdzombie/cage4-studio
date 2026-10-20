"""Single-machine experiment bookkeeping. Each run writes into its own folder."""
from __future__ import annotations
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / 'runs'


def dump_json_atomic(path: Path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', dir=str(path.parent))
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def read_json(path: Path, fallback=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return fallback


def new_run_dir():
    import uuid
    RUNS.mkdir(parents=True, exist_ok=True)
    name = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:6]
    folder = RUNS / name
    folder.mkdir(mode=0o700)
    (folder / 'snapshots').mkdir()
    (folder / 'checkpoints').mkdir()
    return folder


def list_runs():
    if not RUNS.exists():
        return []
    return sorted((x for x in RUNS.iterdir() if x.is_dir() and re.fullmatch(r'\d{8}_\d{6}_[0-9a-f]{6}', x.name)), reverse=True)


def recent_jsonl(path: Path, count: int = 1000):
    from collections import deque
    if not path.exists():
        return []
    with path.open('r', encoding='utf-8') as f:
        lines = deque(f, maxlen=count)
    arr = []
    for line in lines:
        try:
            arr.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a concurrent writer might not have completed its line
    return arr


def status(run_dir: Path):
    return read_json(run_dir / 'status.json', {'state': '尚未启动'})
