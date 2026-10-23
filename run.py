"""Single experiment entry point. PyCharm Run uses DEFAULT_CONFIG."""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('RAY_DEDUP_LOGS', '1')
ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / 'configs/my_algorithm.json'


def main():
    parser = argparse.ArgumentParser(description='CAGE4 training and evaluation')
    commands = parser.add_subparsers(dest='command', required=True)
    train_parser = commands.add_parser('train')
    train_parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    evaluation_parser = commands.add_parser('evaluate')
    evaluation_parser.add_argument('--run', type=Path, required=True)
    evaluation_parser.add_argument('--endpoints', default='final')
    evaluation_parser.add_argument('--seeds', default='54001,54002,54003,54004')
    evaluation_parser.add_argument('--output', type=Path)
    args = parser.parse_args(sys.argv[1:] or ['train'])
    if args.command == 'train':
        from cage4.config import load_config
        from cage4.training import train
        train(load_config(args.config))
    else:
        from cage4.evaluation import evaluate
        evaluate(args.run, [int(x) for x in args.seeds.split(',')], args.endpoints.split(','), args.output)


if __name__ == '__main__':
    main()
