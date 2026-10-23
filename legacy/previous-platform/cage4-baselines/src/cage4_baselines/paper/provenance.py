"""Runtime identities without private paths, credentials or training data."""
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import subprocess

IMPLEMENTATION_EXCLUDES = frozenset(('experiments/run_baseline.py', 'algorithms/my_algorithm.py', 'paper/validation.py'))

SIMULATOR_COMMIT = '8c3c50ca54b176c2de199847944e8dcc035497e3'


def source_digest(root, excludes=()):
    root=Path(root);h=hashlib.sha256()
    for p in sorted(root.rglob('*.py')):
        if '__pycache__' in p.parts or p.relative_to(root).as_posix() in excludes: continue
        h.update(p.relative_to(root).as_posix().encode());h.update(b'\0');h.update(p.read_bytes())
    return h.hexdigest()


def runtime_identity():
    import CybORG
    package=Path(CybORG.__file__).resolve().parent
    root=package.parent
    def git(*args):
        return subprocess.run(['git','-C',str(root),*args],check=True,capture_output=True,text=True,timeout=10).stdout.strip()
    if git('rev-parse','HEAD') != SIMULATOR_COMMIT:
        raise ValueError('Simulator commit differs from the version calibrated for this protocol')
    if git('status','--porcelain','--untracked-files=no'):
        raise ValueError('Simulator tracked source changed; create a new benchmark/calibration instead')
    versions={}
    for name in ('torch','numpy','gym','gymnasium','PyYAML','setuptools'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:pass
    return dict(simulator_commit=SIMULATOR_COMMIT,simulator_sha256=source_digest(package),
                implementation_sha256=source_digest(Path(__file__).resolve().parents[1], IMPLEMENTATION_EXCLUDES),
                python=platform.python_version(),dependencies=versions)
