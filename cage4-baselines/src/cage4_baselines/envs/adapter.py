"""Five-agent CAGE4 API: reset() -> policy inputs; step(actions) -> transition.

Only official Blue observations and action metadata enter PolicyInput.
Researcher telemetry is collected separately by the experiment runner.
"""
from numbers import Integral
import numpy as np
from .factory import BLUE_NAMES, create_env
from .observations import StructuredObservation, name
from ..core.types import PolicyInput


class Cage4Adapter:
    def __init__(self, seed=7629, steps=100, red='FiniteStateRedAgent', observation=None, include_graph=True):
        from CybORG.Agents.Wrappers import BlueFixedActionWrapper
        self.include_graph = include_graph
        self.seed = self._seed(seed)
        if type(steps) is not int or not 10 <= steps <= 1000:
            raise ValueError('steps must be an integer from 10 to 1000')
        self.base = create_env(seed=self.seed, steps=steps, red=red)
        self.env = BlueFixedActionWrapper(self.base, pad_spaces=True)
        commands = sorted({label.replace('[Invalid] ', '').split()[0]
                           for a in BLUE_NAMES for label in self.env.action_labels(a)
                           if not label.startswith('[Padding]')})
        max_hosts = max(len(self.env.hosts(a)) for a in BLUE_NAMES)
        self.encoders = {a: StructuredObservation(self.env.hosts(a), commands, max_hosts,
                                                steps, int(a.rsplit('_',1)[1]), observation) for a in BLUE_NAMES}
        self.step_count = 0
        self.raw = {}
        self.current = {}
        self._started = False
        self._done = False

    @staticmethod
    def _seed(seed):
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError('seed must be a uint32 integer')
        return int(seed)

    def _inputs(self, raw, info):
        result = {}
        for agent, observation in raw.items():
            mask = np.asarray(info[agent]['action_mask'], dtype=bool).copy()
            labels = tuple(map(str, self.env.action_labels(agent)))
            if mask.ndim != 1 or len(mask) != len(labels) or not mask.any():
                raise RuntimeError(f'Invalid official action mask for {agent}')
            # During multi-tick actions the simulator ignores newly chosen actions.
            # Expose only Sleep so ignored choices cannot contaminate PPO likelihoods.
            if name(observation.get('success', 'UNKNOWN')) == 'IN_PROGRESS':
                mask[:] = False
                mask[labels.index('Sleep')] = True
            encoder = self.encoders[agent]
            vector = encoder.encode(observation, self.step_count)
            graph = encoder.graph() if self.include_graph else None
            if graph is not None: graph['action_mask'] = mask.copy()
            result[agent] = PolicyInput(vector, mask, labels, graph)
        self.raw = raw
        self.current = result
        return result

    def reset(self, seed=None):
        """Reset an episode. Omitted seed reuses the constructor seed."""
        episode_seed = self.seed if seed is None else self._seed(seed)
        raw, info = self.env.reset(seed=episode_seed)
        if set(raw) != set(BLUE_NAMES):
            raise RuntimeError('Expected exactly five Blue agents')
        self.step_count = 0
        for a,encoder in self.encoders.items():
            rng_seed = int(np.random.SeedSequence([episode_seed,encoder.agent_index,encoder.config['seed']]).generate_state(1)[0])
            encoder.reset(rng_seed)
        result = self._inputs(raw, info)
        self._started = True
        self._done = False
        return result

    def step(self, actions):
        """Validate all active Blue actions before executing a simulator step."""
        if not self._started or self._done:
            raise RuntimeError('Call reset() before stepping a new or completed episode')
        if not isinstance(actions, dict) or set(actions) != set(self.current):
            raise ValueError('Provide one action for every active Blue agent')
        normalized = {}
        for agent, index in actions.items():
            mask = self.current[agent].mask
            if (isinstance(index, bool) or not isinstance(index, Integral)
                    or index < 0 or index >= len(mask) or not mask[index]):
                raise ValueError(f'Illegal action index: {agent}/{index}')
            normalized[agent] = int(index)
        raw, rewards, terminated, truncated, info = self.env.step(normalized)
        self.step_count += 1
        active = tuple(self.current)
        next_inputs = self._inputs(raw, info)
        self._done = not next_inputs or all(
            terminated.get(agent, False) or truncated.get(agent, False) for agent in active)
        return next_inputs, {a: float(v) for a, v in rewards.items()}, terminated, truncated

    def observation_schema(self):
        return {a:e.schema() for a,e in self.encoders.items()}

    def inputs(self, values):
        """Compatibility for v0.3.0 runners; reset/step now track state themselves."""
        self.current = values
        return values
