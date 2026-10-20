"""在远程 PyCharm 中右键本文件 → Run，即可训练；不需要填写运行参数。"""
from pathlib import Path
import sys

# ===== 日常只改这三项，然后右键运行本文件 =====
ALGORITHM = "mappo"  # ippo / mappo / happo / mappo_lagrangian / macpo / my_algorithm
EPISODES = 2        # 首次先跑 2 回合；正式训练再按实验设计增大
SEED = 101

PROJECT = Path(__file__).resolve().parent
# 显式使用本项目源码，使现有服务器 Python 和独立研究 Python 都可运行。
# 不切换解释器、不启动子进程，因此 PyCharm 的断点调试仍有效。
sys.path[:0] = [str(PROJECT), str(PROJECT / "src"), str(PROJECT.parent / "protocol")]


def main():
    from cage4_baselines.paper import ALGORITHMS
    from cage4_baselines.paper.__main__ import load_config, run
    if ALGORITHM not in (*ALGORITHMS, "my_algorithm"):
        raise ValueError("ALGORITHM 请选择文件顶部列出的算法名")
    if type(EPISODES) is not int or not 1 <= EPISODES <= 50000:
        raise ValueError("EPISODES 必须是 1 到 50000 的整数")
    if type(SEED) is not int or not 1 <= SEED <= 1000:
        raise ValueError("SEED 必须是 1 到 1000 的整数")
    # my_algorithm.py 是基于 MAPPO 的自定义算法接入示例，默认配置从 MAPPO 读取。
    base = "mappo" if ALGORITHM == "my_algorithm" else ALGORITHM
    config = load_config(PROJECT / "configs" / "paper" / (base + ".yaml"))
    config.update(algorithm=ALGORITHM, episodes=EPISODES, seed=SEED,
                  checkpoint_every=min(config["checkpoint_every"], EPISODES))
    custom = None
    if ALGORITHM == "my_algorithm":
        from my_algorithm import MyAlgorithm
        custom = MyAlgorithm
    print(f"开始训练：{ALGORITHM}，{EPISODES} 回合，seed={SEED}", flush=True)
    print("第一个完整回合结束后，网页会出现回报、成本和训练指标。", flush=True)
    path = run(config, Path.home() / "cage4-experiments", Path.home() / "cage4-upstreams",
               PROJECT / "configs" / "safety" / "availability-v1.json", policy_factory=custom)
    print(f"训练完成。实验和权重保存在：{path}", flush=True)


if __name__ == "__main__":
    main()
