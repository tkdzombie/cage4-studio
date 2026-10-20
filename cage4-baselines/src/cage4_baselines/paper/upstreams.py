"""Load external source at exact audited commits, without pip-installing simulators."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import types
from importlib.machinery import ModuleSpec

LOCK = json.loads(Path(__file__).with_name('upstreams.json').read_text())


def command(*args):
    return subprocess.check_output(args, text=True).strip()


def prepare(root):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for name, entry in LOCK.items():
        path = root/name
        if not path.exists():
            subprocess.run(['git', 'init', str(path)], check=True)
            subprocess.run(['git', '-C', str(path), 'remote', 'add', 'origin', entry['url']], check=True)
            subprocess.run(['git', '-C', str(path), 'fetch', '--depth=1', 'origin', entry['commit']], check=True)
            subprocess.run(['git', '-C', str(path), 'checkout', '--detach', 'FETCH_HEAD'], check=True)
        verify(root, name)
    return root


def verify(root, name):
    path = Path(root).expanduser().resolve()/name
    entry = LOCK[name]
    if command('git', '-C', str(path), 'rev-parse', 'HEAD') != entry['commit']:
        raise RuntimeError(f'{name}: unexpected commit; use a separate pinned checkout')
    # Ignore bytecode/untracked files, reject all changes to tracked upstream sources.
    if command('git', '-C', str(path), 'status', '--porcelain', '--untracked-files=no'):
        raise RuntimeError(f'{name}: modified upstream source')
    return path


def activate(root, algorithm):
    name = 'macpo' if algorithm in ('macpo', 'mappo_lagrangian') else 'on-policy'
    path = verify(root, name)
    subdir = {'macpo': 'MACPO', 'mappo_lagrangian': 'MAPPO-Lagrangian'}.get(algorithm, '')
    sys.path.insert(0, str(path/subdir))
    package = {'macpo': 'macpo', 'mappo_lagrangian': 'mappo_lagrangian'}.get(algorithm, 'onpolicy')
    package_path = path/subdir/package
    if package not in sys.modules:
        # Upstream root __init__ eagerly imports unrelated simulator runners and
        # WandB. Load the same algorithm modules as a namespace package instead.
        module = types.ModuleType(package)
        module.__path__ = [str(package_path)]
        module.__package__ = package
        module.__spec__ = ModuleSpec(package, loader=None, is_package=True)
        sys.modules[package] = module
    elif list(sys.modules[package].__path__) != [str(package_path)]:
        raise RuntimeError(f'{package}: already imported from an unverified source')
    files = command('git', '-C', str(path), 'ls-files', '*.py', 'LICENSE*').splitlines()
    digest = hashlib.sha256()
    for file in sorted(files):
        digest.update(file.encode()); digest.update((path/file).read_bytes())
    return dict(name=name, **LOCK[name], source_sha256=digest.hexdigest(),
                loader='namespace root; original algorithm modules, no upstream runners')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    args = p.parse_args()
    print(prepare(args.root))


if __name__ == '__main__':
    main()
