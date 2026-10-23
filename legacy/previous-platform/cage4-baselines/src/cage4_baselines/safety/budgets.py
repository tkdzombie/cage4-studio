"""Load historical calibrated budgets or explicitly specified research targets."""
import json
import math
from pathlib import Path
from .costs import COST_VERSION


def read_budget(path, steps, smoke=False, level='main'):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if data.get('cost_version') != COST_VERSION:
        raise ValueError('Incompatible cost version')
    schema = data.get('schema')
    if schema == 'cage4.budget-calibration/1':
        if data.get('status') != 'calibrated':
            raise ValueError('Budget did not pass independent reference validation')
    elif schema == 'cage4.budget-target/1':
        # A target is a research design, not a claim that a feasible policy exists.
        if data.get('status') != 'specified' or data.get('feasibility') != 'not_established':
            raise ValueError('Invalid research target status')
        if not data.get('design') or not data.get('reference_calibration_sha256'):
            raise ValueError('Research target must document its design and reference')
    else:
        raise ValueError('Incompatible budget schema')
    if not smoke and steps != data['configured_steps']:
        raise ValueError('Horizon differs from calibration; recalibrate or use --smoke for interface tests')
    if level not in ('main', 'strict', 'loose'):
        raise ValueError('Invalid budget level')
    rates = [data['budgets'][k+'_rate'] for k in ('strict', 'main', 'loose')]
    if any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x <= 1 for x in rates):
        raise ValueError('Invalid budget rate')
    if rates != sorted(rates):
        raise ValueError('Budget levels must be ordered strict <= main <= loose')
    return data, data['budgets'][level+'_rate']
