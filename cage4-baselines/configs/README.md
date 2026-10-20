# 实验配置

| 配置 | 用途 | 默认 Episode 数 |
|---|---|---:|
| quickstart/actor_critic.yaml | 短程一键基线演示 | 2（评估为 1 / seed） |
| train/actor_critic.yaml | Actor-Critic 训练 | 30 |
| evaluate/actor_critic.yaml | 加载 checkpoint 后评估 | 5 / seed |
| baselines/random.yaml | Random 对照评估 | 5 / seed |
| baselines/sleep.yaml | Sleep 对照评估 | 5 / seed |

复制最接近的模板，使用明确名称，例如 train/ac_lr0001_seed42.yaml。算法相关参数、仿真 seed/steps/red、记录频率都写在同一份有效 YAML 中；这是简单的完整配置文件，不支持隐式继承或 YAML 拼接。schema 和范围校验在 src/cage4_baselines/config.py，未知参数会报错。

配置只决定本次新启动的实验，不会改变已有进程。每个 run 自动保存规范化 config.json。运行路径、checkpoint 与评估 seeds 由 CLI 显式传入，不要把个人绝对路径、密钥、SSH 设置放进公开 YAML。

平铺的 actor_critic.yaml / random.yaml / sleep.yaml 保留 v0.3.0 的原始参数以兼容历史命令；新增实验请使用上表分目录配置。历史 Random/Sleep 模板仍是 30 Episode，新的对照评估模板是 5 Episode。不要混用后误判实验量。
