# 03｜实验与可视化的正确使用方式

## 一、最短的完整演示流程

1. 在 Mac Studio 执行 `bash check.sh`（必须看到 `SMOKE TEST PASS`），然后 `bash start.sh`。
2. 浏览器打开控制台；侧边栏配置 `Seed=7629`、`Steps=10`、`Episodes=2`、`Device=cpu`、`Learning Rate=0.0003`。
3. 切到“训练监控”，点击“开始训练”。这是一个五蓝队独立 Actor-Critic **研究原型**。
4. 页面会显示运行状态与 Episode Return；等待第一轮完成才有第一条 episode 级曲线。
5. 切到“训练中的网络拓扑”，观察 CAGE 4 生成的主机、接口 `data_links`、Red Session。可以选择过滤子网，以及本 Episode 的保存步回放。
6. 切到“单步仿真调试”，运行独立的 SleepAgent 对照环境，可以单步/十步推进、检查官方 observation。
7. “运行记录”展示日志与指标；TensorBoard 单独访问 `http://127.0.0.1:6006`。

## 二、各标签页的作用

| 标签页 | 用途 | 注意 |
|---|---|---|
| 训练监控 | 启动/停止训练，查看团队与各蓝队回报、轮数和日志 | 训练曲线是真实算法采样，但不保证上升/收敛 |
| 训练中的网络拓扑 | 展示真实接口链路、主机、红蓝 Session、快照及行动 | 红色高亮只表示 Red Session，不能直接代表真实网络攻击成功率 |
| 单步仿真调试 | 重置官方 SleepAgent 基线、推进仿真、观察动作和奖励 | 与训练进程**独立**，数据不能混为一轮训练 |
| 运行记录与说明 | 查看配置、最近日志、导出指标 | Checkpoint 保存在服务器 runs/ 里 |

## 三、日志与模型输出

每次训练在 `runs/日期时间_随机标识/` 产生：

```text
config.json          实验配置
status.json          当前状态
process.json         训练子进程 PID
train.log            训练标准输出和报错
metrics.jsonl        各 Episode 回报、loss、步数
actions.jsonl        动作与奖励时间序列
topology_latest.json 最近网络真值快照
snapshots/           部分时间步的拓扑快照
checkpoints/         模型和优化器权重
tensorboard/         TensorBoard 事件文件
policy_spaces.json   官方各蓝队观测/动作维度
```

停止训练在“训练监控”选择正确 Run 后点击“请求停止训练”，会在训练进程下次检查时安全结束；终止浏览器、关闭 SSH 隧道不等于终止训练。

## 四、研究有效性边界

- 训练模型只能读取 `BlueFlatWrapper` 的允许 Observation、action mask；研究者可视化读取的完整 `state` 是 **Ground Truth**，不能泄漏到策略输入。
- 节点间连线来自 CAGE4 官方主机 `interfaces[*].data_links`，不表示即时可达性或所有网络防火墙 ACL 状态。
- 内置五蓝队策略是 on-policy Actor-Critic/REINFORCE with value baseline 的工程基线，**不是 PPO / MAPPO / IPPO / 正式 CAGE4 标准评估程序**。
- `team_return` 是五蓝队逐 Episode 回报之和。正式研究需要多随机种子、独立评估、基线对照、指标定义和配置/代码 commit 归档。
- M3 Ultra 的 MPS 可以加速部分 PyTorch 算子，但环境仿真主要在 CPU。建议小规模先 CPU 稳定运行，再试 MPS。

## 五、正式实验建议

先做“2 Episodes × 10 Steps”的小样测试，确认 UI、训练日志、网络快照和模型权重都真实生成；再用 30×100 或更多训练，最后进行跨 Seed 的独立评估。长时训练前备份 `runs/`、`installed-revision.txt`、`installed-pip-freeze.txt`，不要随意升级库版本。
