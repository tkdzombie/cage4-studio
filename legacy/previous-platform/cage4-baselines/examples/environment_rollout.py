"""Run from any directory using the installed research interpreter.

Example: python examples/environment_rollout.py --algorithm random --steps 20
"""
import argparse
import json
from cage4_baselines.envs.adapter import Cage4Adapter
from cage4_baselines.algorithms import create


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', choices=['random','sleep'], default='random')
    parser.add_argument('--seed', type=int, default=7629)
    parser.add_argument('--steps', type=int, default=20)
    args = parser.parse_args()
    env = Cage4Adapter(seed=args.seed, steps=args.steps)
    inputs = env.reset()
    policy = create(args.algorithm, inputs, {'seed':args.seed, 'plugins':[]})
    print('Blue observation dimensions:', {a:len(x.observation) for a,x in inputs.items()})
    for step in range(1,args.steps+1):
        actions = policy.act(inputs)
        inputs,rewards,terminated,truncated = env.step(actions)
        print(json.dumps({'step':step,'actions':actions,'rewards':rewards}))
        if not inputs or all(terminated.get(a,False) or truncated.get(a,False) for a in actions):
            break


if __name__ == '__main__':
    main()
