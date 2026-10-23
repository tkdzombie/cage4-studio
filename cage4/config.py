"""One validated experiment configuration shared by CLI, training and snapshots."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]

@dataclass
class TrainConfig:
    run: str
    algorithm: str = 'ppo'
    initialization: str = 'scratch'
    seed: int = 321
    iterations: int = 150
    workers: int = 24
    batch: int = 8192
    minibatch: int = 2048
    epochs: int = 10
    lr: float = .0003
    gae_lambda: float = 1.0
    reward_scale: float = .01
    protocol: str = 'flat-full-actions-history-correct'
    checkpoint_every: int = 10
    evaluation_seeds: list = field(default_factory=lambda: [54001, 54002, 54003, 54004])

    def __post_init__(self):
        if self.algorithm not in ('ppo', 'my_algorithm'):
            raise ValueError('algorithm must be ppo or my_algorithm')
        if self.initialization not in ('scratch', 'author'):
            raise ValueError('initialization must be scratch or author')
        if self.protocol != 'flat-full-actions-history-correct':
            raise ValueError('Use the fixed CAGE4 protocol for the primary comparison')
        if any(x < 1 for x in (self.iterations, self.batch, self.minibatch, self.epochs, self.checkpoint_every)):
            raise ValueError('Training budgets and intervals must be positive')
        if self.workers < 0 or self.lr <= 0 or self.reward_scale <= 0:
            raise ValueError('Invalid worker count, learning rate or reward scale')
        if not self.evaluation_seeds:
            raise ValueError('At least one evaluation scenario is required')

    @property
    def output(self):
        path = Path(self.run)
        return path.resolve() if path.is_absolute() else ROOT / path

    def to_dict(self):
        return asdict(self)


def load_config(path):
    return TrainConfig(**json.loads(Path(path).read_text()))
