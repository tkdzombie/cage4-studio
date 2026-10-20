"""One command executes real environment, training and comparable evaluations."""
import hashlib
from pathlib import Path
import tempfile
import traceback
from cage4_protocol.store import atomic_json, read_json
from ..config import load
from ..envs.smoke import smoke
from ..envs.factory import BLUE_NAMES
from .runner import run


def _finished(path):
    state = read_json(path/'status.json', {}).get('state')
    if state != 'finished':
        raise InterruptedError(f'Run {path.name} ended with {state}; starter suite aborted')
    return path


def quickstart(runs_root, config_path=None, seeds=None):
    import torch
    root = Path(runs_root).expanduser().resolve()
    seeds = [1001, 1002, 1003] if seeds is None else seeds
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int or not 0 <= s < 2**32 for s in seeds):
        raise ValueError('Use distinct uint32 evaluation seeds')
    profile = config_path or Path(__file__).resolve().parents[3]/'configs/quickstart/actor_critic.yaml'
    config = load(profile)
    if config['algorithm'] != 'actor_critic' or config['episodes'] < 2:
        raise ValueError('Quickstart requires Actor-Critic and at least two training episodes')
    config['checkpoint_every'] = 1
    suites = root/'starter_suites'
    suites.mkdir(parents=True, exist_ok=True)
    suite = Path(tempfile.mkdtemp(prefix='quickstart_', dir=suites))
    report = {'schema': 'cage4.starter/1', 'state': 'running', 'evaluation_seeds': seeds,
              'purpose': 'short runnable baseline demonstration, not a performance claim', 'runs': {}}
    atomic_json(suite/'report.json', report)
    try:
        print('[1/5] Real CAGE4 environment smoke episode', flush=True)
        report['environment'] = smoke(config['seed'], config['steps'])
        print('[2/5] Train five independent Actor-Critic policies', flush=True)
        trained = _finished(run(config, root))
        report['runs']['train'] = str(trained)
        checkpoints = sorted((trained/'checkpoints').glob('*.pt'))
        first = torch.load(checkpoints[0], map_location='cpu', weights_only=True)
        last = torch.load(checkpoints[-1], map_location='cpu', weights_only=True)
        changed = {agent: any(not torch.equal(value, last['policy']['models'][agent][key])
                             for key, value in model.items())
                   for agent, model in first['policy']['models'].items()}
        if set(changed) != set(BLUE_NAMES) or not all(changed.values()):
            raise RuntimeError(f'Expected all five models to update: {changed}')
        report['weights_changed'] = changed
        report['checkpoint'] = str(checkpoints[-1])
        checkpoint_hash = hashlib.sha256(checkpoints[-1].read_bytes()).hexdigest()
        report['checkpoint_sha256'] = checkpoint_hash
        evaluation_config = {**config, 'episodes': 1}
        report['comparison'] = {}
        for number, algorithm in enumerate(('actor_critic', 'random', 'sleep'), 3):
            print(f'[{number}/5] Evaluate {algorithm} on seeds {seeds}', flush=True)
            checkpoint = checkpoints[-1] if algorithm == 'actor_critic' else None
            result = _finished(run({**evaluation_config, 'algorithm': algorithm}, root,
                                   'evaluate', checkpoint, seeds))
            report['runs'][algorithm] = str(result)
            report['comparison'][algorithm] = read_json(result/'evaluation.json')
            atomic_json(suite/'report.json', report)
        if hashlib.sha256(checkpoints[-1].read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError('Evaluation altered the saved checkpoint')
        report.update(state='finished', checkpoint_unchanged_by_evaluation=True)
        atomic_json(suite/'report.json', report)
        print(f'PASS: environment, five model updates, and {len(seeds)}-seed baseline evaluations', flush=True)
        return suite/'report.json'
    except BaseException as exc:
        report.update(state='stopped' if isinstance(exc, (KeyboardInterrupt, InterruptedError)) else 'error',
                      error=str(exc), traceback=traceback.format_exc())
        atomic_json(suite/'report.json', report)
        raise
