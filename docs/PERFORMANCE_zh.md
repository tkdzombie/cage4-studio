# Mac Studio MAPPO 多核训练

checkpoint默认每50个episode保存一次，结束时额外保存最后一次。运行入口仍是 `src/cage4_baselines/experiments/run_baseline.py`，Run/Debug无需额外参数。已有checkpoint删除不会恢复，开始新实验从头训练。

服务器实际有20性能核+8效率核、96GB统一内存。旧实现一次只推进一个Python环境，并将Torch限制为1线程。完整回合的cProfile显示主要耗时在官方CybORG仿真和奖励计算，尤其get_true_state/Observation对象构造；这部分受Python GIL限制，提高Torch线程数无法替代独立环境进程。剖析本身有额外开销，因此不能用cProfile墙钟与未剖析训练直接计算加速倍数。

## 执行方式

- `num_envs`：独立环境进程数；每个子进程Torch/BLAS限制为1线程。
- `torch_threads`：主进程模型计算线程数；主要影响初始化和更新。
- `rollout_episodes`：一个共同策略下的梯度批次回合数，五算法profile统一16（以前为8）。500配置步实际499次transition，通常每批7984次环境交互。
- `checkpoint_every: 50`：独立于梯度批次；不会为了保存50回合checkpoint，改成2回合小批次更新。

子进程只推演环境、用冻结actor/critic采样并收集旧logp/value/reward/cost。主进程收齐完整批次后统一更新。每个回合seed控制独立采样随机流，结果按episode编号合并，避免完成顺序导致数据错配。单进程和spawn多进程的逐步观测、动作、logp和cost有实机一致性测试。性能参数/批次大小、随机数协议均保存到配置和来源记录，旧版实验与新版不可混称同一次训练。

每个worker写独占episode帧目录，完整回放不删减；只读网页仍能读取相同原子JSON协议。主进程维护唯一状态、指标和checkpoint。自定义my_algorithm入口暂使用单进程，保留普通PyCharm断点；MAPPO可按需要将num_envs设1调试，再恢复并行。spawn子进程断点需PyCharm启用多进程调试，单步调试的耗时不能拿来衡量正常训练速度。

MAPPO不会向actor提供Ground Truth、Green业务明细或全网拓扑；图sidecar不送入作者MLP，因此只省去未使用的图构造，不改变2345维实际策略输入。未修改官方CybORG源码、奖励、动作时长或观测编码字段。

## 实测方法

以下在远程Mac Studio终端执行，输出放临时目录，避免占用最近5次正式实验名额：

```bash
cd "$HOME/cage4-research"
export PYTHONPATH="$PWD/protocol:$PWD/cage4-baselines/src:$PWD/server/src"
"$HOME/cage4-studio/.conda-env/bin/python" ops/benchmark_mappo.py \
  --output /tmp/cage4-benchmark --upstreams "$HOME/cage4-upstreams" \
  --workers 1 4 8 16 --threads 1 --episodes 16 --batch 16
```

比较时模型仍为256双隐藏层、完整500配置步、同样16回合梯度批次、相同seed和超参数。默认关闭回放以隔离计算吞吐；加`--replay`核对完整记录的速度。输出同时给出包含spawn/初始化/最终checkpoint的端到端速度，以及采样和模型更新时间；短任务可能因spawn启动开销反而更慢。不要声称28进程必然最快。

网页显示并行环境数、模型线程数、完成回合、阶段及总体环境步/秒；回报同时画原始值与50回合滑动平均。CPU占用按整个进程树判断，不只看主训练器单个PID。采样与更新是分阶段执行，所有核心不会在每一时刻都满载。

参考：[Python multiprocessing](https://docs.python.org/3.10/library/multiprocessing.html)明确建议macOS使用spawn；[PyTorch multiprocessing](https://docs.pytorch.org/docs/stable/notes/multiprocessing.html)讨论进程与线程过量争用。


## 本机实测（2026-10-20）

单次16完整episode、7984次环境交互，256双隐藏层、PPO10轮、批次16，计算测试关闭回放：

| 环境进程 | 主模型线程 | 端到端秒数 | 环境步/秒 | 相对单环境 |
|---:|---:|---:|---:|---:|
| 1 | 1 | 156.77 | 50.93 | 1.00× |
| 4 | 1 | 63.72 | 125.29 | 2.46× |
| 8 | 1 | 48.09 | 166.03 | 3.26× |
| 16 | 1 | 36.63 | 217.95 | 4.28× |
| 16 | 4 | 31.83 | 250.81 | 4.93× |
| 16 | 8 | 28.62 | 278.99 | 5.48× |

主进程模型更新耗时从约12.50秒（1线程参考）变为约7.63秒（8线程）；16环境采样约15.42秒。其余为启动、来源校验、模型初始化与保存等。每个配置仅一次诊断测量，包含正常系统噪声，不是多次重复的论文性能统计，亦不是收敛结果。默认采用本次已测组合中最快的16环境/8模型线程，未声称28环境更快。

同1模型线程的四种进程数得到完全一致的16个回合回报，最终actor/critic最大参数差为0。模型线程数改变可能引入浮点归约差异，不要求跨线程位级一致。正式训练默认仍开启完整回放；其实际速度由performance.json/网页实时显示，不能直接套用关闭回放的表格数字。
