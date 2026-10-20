"""Public algorithm registry and built-in implementations (Torch imports are lazy)."""
from .registry import REGISTRY, create, register
from .random import RandomPolicy
from .sleep import SleepPolicy
from .actor_critic import ActorCriticPolicy
__all__ = ['REGISTRY','create','register','RandomPolicy','SleepPolicy','ActorCriticPolicy']
