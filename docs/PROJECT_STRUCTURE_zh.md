# 项目架构与文件放置规范

## 1. 仓库与三个工程

```text
cage4-studio/                         # Git 仓库根目录
├── cage4-baselines/                  # PyCharm 打开的研究项目
├── server/src/cage4_server/          # 只读网页与记录解析
├── client/                          # 本地 macOS SSH 客户端
├── protocol/                        # 三方约定的 cage4.run/1 记录协议
├── ops/                             # 运维、验证、打包、便捷入口
├── tests/                           # 协议、Server 读取与运维测试
├── docs/                            # 使用手册、架构、安装、协议
└── legacy/                          # 历史代码归档，不作为当前入口
```

工程内按职责组织，而不是按“所有 Python 文件平铺”或“网页里包含训练”组织。开发仓库和部署目录分开：研究工作副本可以改代码；Server 固定发布目录保持稳定。依赖环境和实验输出放在源码外。

## 2. 研究项目内部

| 路径 | 责任 | 常见修改 |
|---|---|---|
| pyproject.toml | Python 包、直接依赖、console command | 新增必要依赖、更新包版本 |
| environments/ | 解释器安装说明与经过验证的版本参考 | 记录兼容性，不放实际虚拟环境 |
| configs/train/ | 完整训练 YAML | 训练 seed、lr、episodes、steps |
| configs/evaluate/ | checkpoint 评估 YAML | 每个 seed 的评估次数、环境参数 |
| configs/baselines/ | Random/Sleep 对照 YAML | 与待评估模型保持环境一致 |
| src/cage4_baselines/core/types.py | PolicyInput 数据契约 | 经过设计后扩展输入字段 |
| src/cage4_baselines/envs/factory.py | 创建官方 CybORG 场景 | 环境生成与受支持红队类型 |
| src/cage4_baselines/envs/adapter.py | reset/step、固定动作空间适配 | 适配官方 API 变化 |
| src/cage4_baselines/envs/observations.py | 只编码 Blue 自身 observation | 新特征编码与版本号 |
| src/cage4_baselines/algorithms/base.py | 算法方法契约 | 插件 API 设计 |
| src/cage4_baselines/algorithms/registry.py | 名称注册、插件加载 | 新算法名称与显式插件 |
| src/cage4_baselines/algorithms/actor_critic.py | 五策略 Actor-Critic | 模型结构、损失、优化器 |
| src/cage4_baselines/algorithms/random.py | Random 对照策略 | 合法掩码内随机选择 |
| src/cage4_baselines/algorithms/sleep.py | Sleep 对照策略 | 使用官方 Sleep 动作 |
| src/cage4_baselines/experiments/runner.py | 实验生命周期 | train/evaluate、seed、checkpoint、状态 |
| src/cage4_baselines/recording/telemetry.py | 研究者遥测 | 拓扑、红队 Session、动作与原始观测 |
| src/cage4_baselines/config.py | YAML 加载和字段范围检查 | 新超参数必须同步这里 |
| src/cage4_baselines/__main__.py | 统一命令入口 | 通常无需修改 |
| tests/ | 研究层测试 | 算法、配置、输入隔离、兼容入口 |
| .run/ | PyCharm 共享运行配置 | 常用训练/对照命令 |
| artifacts/ | 可选的临时输出 | 默认被 Git 忽略，正式结果推荐放项目外 |

## 3. “环境”分别指什么

**仿真环境代码**放 envs/。上游 CybORG 仍在外部固定源码目录，是第三方依赖；不把它再复制一份到 envs/。环境 seed、时长 steps、red 类型是实验参数，放 YAML；底层环境生成/step API 的变化才改 Python。

**Python 软件环境**是专用研究解释器，通常位于 ~/cage4-research-env；不能放进 Git。安装说明与依赖参考放 environments/，包直接依赖由 pyproject.toml 声明。版本参考不是完整 lock，不能承诺空白机器一键重建。

**部署环境**是 Server 的固定版本目录及 launchd 定义；由 ops/service.py 管理，不由算法文件或训练 YAML 管理。

## 4. 依赖方向和观测边界

```text
CLI / PyCharm
    → experiments.runner
        → envs → 官方 CybORG（研究进程）
        → algorithms ← core.PolicyInput（Blue observation + 官方动作掩码）
        → recording → Ground Truth 遥测（仅供研究者）
        → protocol → 原子 run 文件

Server → protocol / run 文件 → 浏览器图表与回放
Client → SSH 转发 → Server 网页
```

算法不能导入 envs、recording、CybORG 或直接读取 run 中的拓扑/红队信息。核心算法只接收 PolicyInput；Ground Truth 遥测由 runner 单独调用。测试会检查内置算法没有引入这些模块。它是工程依赖与数据流边界，不是对恶意插件的进程沙箱。

Server 不导入训练代码，不加载 .pt；Client 不创建训练进程、不负责启动 PyCharm。修改新算法不需要修改网页或重建客户端。

## 5. 配置和结果规范

YAML 为显式完整配置，不做隐式继承。新增文件按目的命名：`train/ac_lr0001_seed42.yaml`。不要把密码、SSH 信息或个人绝对路径写入公共 YAML。记录目录用 --runs 显式指定。

正式实验输出推荐 ~/cage4-experiments；具体 run 内的 config/metrics/frames/checkpoints/status/tensorboard 结构见 PROTOCOL.md。不要把输出、checkpoint、.conda-env、.venv、第三方源码或 IDE 私人设置提交 Git。

## 6. 兼容与迁移

v0.3.0 的导入路径 adapter.py、engine.py、observations.py、runner.py、telemetry.py 保留为薄兼容入口，真实实现只有对应子包中的一份。`python -m cage4_baselines.runner` 仍能运行；新命令优先用 `python -m cage4_baselines`。

平铺的旧 YAML 为原参数兼容副本，明确不作为新模板；新增实验只使用分目录配置。旧 Random/Sleep YAML 是 30 Episode，新对照评估模板是 5 Episode。两者不要混淆。

本次重构没有改变 Actor-Critic 数学实现或 run schema，也没有启用 IPPO/MAPPO。已保存的 v0.3.0 新编码 checkpoint 继续按原有张量容器格式读取；更早 BlueFlat 编码权重仍不兼容。Server 固定 v0.3.0 部署无需因研究目录整理而重装。
