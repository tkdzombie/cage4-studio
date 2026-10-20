"""Official CAGE 4 environment adapter. Imports are deliberately lazy."""

RED_NAMES = ('FiniteStateRedAgent', 'DiscoveryFSRed')
BLUE_NAMES = tuple(f'blue_agent_{i}' for i in range(5))


def create_env(*, seed: int, steps: int, red: str = 'FiniteStateRedAgent'):
    from CybORG import CybORG
    from CybORG.Agents import (
        DiscoveryFSRed, EnterpriseGreenAgent, FiniteStateRedAgent, SleepAgent,
    )
    from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

    reds = {'FiniteStateRedAgent': FiniteStateRedAgent,
            'DiscoveryFSRed': DiscoveryFSRed}
    if red not in reds:
        raise ValueError(f'Unsupported red agent: {red}')
    gen = EnterpriseScenarioGenerator(
        blue_agent_class=SleepAgent, green_agent_class=EnterpriseGreenAgent,
        red_agent_class=reds[red], steps=int(steps),
    )
    return CybORG(scenario_generator=gen, seed=int(seed))


def blue_reward(env) -> float:
    """Aggregate blue team reward from the most recent unwrapped step."""
    from numbers import Real
    rewards = env.get_rewards().get('Blue', {})
    if isinstance(rewards, dict):
        return sum(float(x) for x in rewards.values() if isinstance(x, Real))
    return float(rewards) if isinstance(rewards, Real) else 0.0


def action_name(env, agent):
    try:
        return str(env.get_last_action(agent))
    except (KeyError, IndexError, AttributeError, ValueError):
        return '—'
