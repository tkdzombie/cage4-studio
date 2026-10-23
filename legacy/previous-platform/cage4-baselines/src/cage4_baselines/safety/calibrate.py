"""Independent fixed-policy budget calibration; never uses learned-policy scores."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from cage4_protocol.store import atomic_json
from .costs import COST_VERSION, measure, summarize

CALIBRATION_SEEDS = list(range(41001, 41021))
VALIDATION_SEEDS = list(range(42001, 42021))


def propose(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) < 20 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Need at least 20 finite episode costs in [0, 1]')
    # Student t, df >= 19, one-sided 95%; 1.729 is conservative for n > 20.
    upper = float(values.mean() + 1.729 * values.std(ddof=1) / np.sqrt(len(values)))
    main = min(1., max(float(np.quantile(values, .75)), upper, .001))
    return dict(main_rate=main, strict_rate=min(main, float(np.quantile(values, .5))),
                loose_rate=max(main, float(np.quantile(values, .9))),
                reference_mean=float(values.mean()), reference_mean_upper95_t=upper,
                rule='max(q75, mean + 1.729*sample_sd/sqrt(n), 0.001), clipped to 1',
                confidence_note='Approximate one-sided t interval, not a distribution-free guarantee')


def collect(seeds, steps):
    from ..envs.adapter import Cage4Adapter
    from ..algorithms.sleep import SleepPolicy
    env = Cage4Adapter(steps=steps)
    rows = []
    for seed in seeds:
        inputs = env.reset(seed)
        policy = SleepPolicy(inputs, {'seed': seed})
        events = []
        for _ in range(steps):
            actions = policy.act(inputs)
            inputs, rewards, terminated, truncated = env.step(actions)
            events.append(measure(env.base))
            if not inputs or all(terminated.get(a, False) or truncated.get(a, False) for a in rewards):
                break
        row = dict(seed=seed, **summarize(events))
        rows.append(row)
        print(json.dumps(row), flush=True)
    return rows


def calibrate(output, steps=500):
    calibration = collect(CALIBRATION_SEEDS, steps)
    budgets = propose([r['mean_cost'] for r in calibration])
    validation = collect(VALIDATION_SEEDS, steps)
    val_mean = float(np.mean([r['mean_cost'] for r in validation]))
    from CybORG import __file__ as cyborg_file
    reward_file = Path(cyborg_file).parent/'Shared'/'BlueRewardMachine.py'
    result = dict(schema='cage4.budget-calibration/1', cost_version=COST_VERSION,
                  configured_steps=steps, red='FiniteStateRedAgent', reference='sleep',
                  calibration=calibration, validation=validation, budgets=budgets,
                  reward_source_sha256=hashlib.sha256(reward_file.read_bytes()).hexdigest(),
                  validation_mean=val_mean, validation_mean_feasible=val_mean <= budgets['main_rate'],
                  status='calibrated' if val_mean <= budgets['main_rate'] else 'validation_failed',
                  warning='Research benchmark budget, not an official CAGE4 SLA or per-episode guarantee')
    atomic_json(Path(output), result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=500)
    args = p.parse_args()
    print(json.dumps(calibrate(args.output, args.steps)['budgets'], indent=2))


if __name__ == '__main__':
    main()
