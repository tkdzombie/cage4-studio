"""PyCharm: edit these settings, then Run. Uses this project's interpreter."""
from pathlib import Path
import os
import subprocess
import sys

ALGORITHM = 'my_algorithm'  # 'ppo' for baseline; template is initially identical
INITIALIZATION = 'author'  # 'author' or 'scratch'; match both comparison arms
RUN_NAME = 'my-algorithm-author-s321-v1'  # existing runs are never overwritten
SEED = 321
ITERATIONS = 30
WORKERS = 24

ROOT = Path(__file__).resolve().parents[1]

def command():
    if INITIALIZATION not in ('author', 'scratch'):
        raise ValueError('INITIALIZATION must be author or scratch')
    args = [sys.executable, '-u', str(ROOT/'train.py'), '--level', 'flat',
            '--algorithm', ALGORITHM, '--run', str(ROOT/'runs'/RUN_NAME),
            '--seed', str(SEED), '--workers', str(WORKERS), '--batch', '8192',
            '--iterations', str(ITERATIONS), '--reward-scale', '.01',
            '--protocol', 'flat-full-actions-history-correct', '--gae-lambda', '1',
            '--lr', '.00005' if INITIALIZATION == 'author' else '.0003',
            '--epochs', '30' if INITIALIZATION == 'author' else '10', '--minibatch', '2048']
    if INITIALIZATION == 'author':
        upstream = Path(os.environ.get('CAGE4_ROOT', ROOT))
        args += ['--warm-start', str(upstream/'Hierarchical-MARL/marl-1policy/models/train_marl/iter_199'),
                 '--warm-start-value-scale', '.01']
    return args

if __name__ == '__main__':
    env = dict(os.environ, CAGE4_ROOT=os.environ.get('CAGE4_ROOT', str(ROOT)),
               OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    raise SystemExit(subprocess.call(command(), cwd=ROOT, env=env))
