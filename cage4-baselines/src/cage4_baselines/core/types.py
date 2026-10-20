from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class PolicyInput:
    observation: np.ndarray
    mask: np.ndarray
    labels: tuple
    graph: dict | None = None
