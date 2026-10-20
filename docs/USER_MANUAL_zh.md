> PyCharm日常启动请先看 [简明使用说明](START_SIMPLE_zh.md)：打开 `run_baseline.py` 右键 Run 即可，无需手动配置模块与参数。下面保留完整管理与批量实验说明。

# CAGE4 Studio 科研系统使用说明书

适用范围：已有 macOS / CybORG / CAGE4 Studio 的本地客户端、远程研究项目及 Server。安装升级细节见 [安装与回滚](INSTALL_zh.md)，找文件见 [项目架构](PROJECT_STRUCTURE_zh.md)。本文所有命令均说明执行位置。主机名、SSH 用户和个人绝对路径按自己的环境填写；公共说明书不保存私人连接信息。

首次先运行 [基础环境和 Baseline 跑通指南](BASELINES_START_zh.md) 中的一键命令，即可得到真实训练与对照实验。

## 1. 系统如何工作

你在远程 PyCharm 中写算法，研究进程运行 CybORG 并产生实验文件；Server 像实验观测台一样只读这些文件；本地 Client.app 建立安全隧道并打开网页。

| 使用位置 | 工作内容 | 不承担的工作 |
|---|---|---|
| 本地 MacBook 的 Toolbox / PyCharm 客户端 | 连接远程开发项目、编辑与运行代码 | 不需要另装第二套本地 PyCharm |
| 远程研究项目 cage4-baselines | 算法、仿真适配、配置、训练、评估、写记录 | 不在网页中调度研究任务 |
| 远程 Studio Server | 曲线、拓扑、红队记录、回放、对比 | 不改变训练参数、不加载模型权重 |
| 本地 CAGE4 Studio Client.app | SSH 端口转发、打开浏览器 | 不启动 PyCharm、不决定算法 |

关闭网页不会停止训练。修改 YAML 不会影响已启动实验。打开客户端不会自动开始新实验。

## 2. 先认识目录

推荐远程目录：

```text
~/cage4-research/                       # 可修改的 Git 工作副本
  cage4-baselines/                      # Toolbox / PyCharm 打开这里
~/cage4-research-env/                   # 研究解释器，不进入 Git
~/cage4-experiments/                    # 所有新实验记录，不进入 Git
~/cage4-studio-releases/<版本>/          # 固定 Server 发布目录
~/cage4-studio/                         # 原安装与依赖，升级后保留
~/cage4-studio-backups/                 # plist 与旧源码备份
```

研究项目里的 environments/ 是安装说明，不是放整套 Python 的目录。envs/ 是仿真接口 Python 代码；configs/ 是实验参数 YAML；algorithms/ 是真正的算法实现。详见架构文档的文件职责表。

## 3. 每天如何开始（本地 MacBook）

1. 确认 Mac Studio 已登录相应 macOS 用户，网络和 SSH 可用。现有 Server 为用户级 LaunchAgent，登录后启动。
2. 双击配置好的 CAGE4 Studio Client.app，打开自动建立的本地网页。新客户端默认使用本地 18501，远程仍是 8501。
3. 需要改算法时，在 Toolbox → Remote Development → SSH 打开远程 `~/cage4-research/cage4-baselines`。
4. PyCharm 解释器选择远程 `~/cage4-research-env/bin/python`。它已经安装研究包；无需让本地 Python 承担仿真。
5. 在网页选择 run 查看结果。网页显示的是已写入记录，不是 PyCharm 当前编辑器内容。

Toolbox 的版本可能需要手动创建运行配置：Python module 选择 cage4_baselines，工作目录为 cage4-baselines，参数使用后文示例。参数框不经过 shell，请填写远程绝对输出路径，或使用共享配置的路径宏；不要直接依赖终端中的 $HOME 或 ~ 展开。本项目提供 Train Actor Critic、Evaluate Random、Evaluate Sleep 三个共享配置；Actor-Critic 评估必须由你指定真实 checkpoint。

## 4. 训练一次实验（远程 Mac Studio 终端）

先进入研究项目；以下命令也可在 PyCharm 的远程 Terminal 执行：

```bash
cd "$HOME/cage4-research/cage4-baselines"
cp configs/train/actor_critic.yaml configs/train/ac_seed42.yaml
```

编辑新 YAML，例如 seed: 42、episodes: 30、steps: 100、lr: 0.0003。确认后执行：

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines train \
  --config configs/train/ac_seed42.yaml \
  --runs "$HOME/cage4-experiments"
```

也可以从仓库根目录使用便捷脚本：

```bash
cd "$HOME/cage4-research"
export CAGE4_PYTHON="$HOME/cage4-research-env/bin/python"
bash ops/research.sh train \
  --config cage4-baselines/configs/train/actor_critic.yaml \
  --runs "$HOME/cage4-experiments"
```

脚本路径和 YAML 相对路径取决于工作目录；不要把这两组命令的工作目录混在一起。训练结束输出本次 run 的真实目录。训练中可在记录根目录查找最新 run 的 status.json；网页会定时刷新。

Actor-Critic 是真实五策略 REINFORCE + value baseline，每轮进行梯度更新。它是可运行研究参考，不是 PPO；不能预设奖励必然收敛。IPPO/MAPPO 尚未实现。

## 5. 看懂一个 run

| 文件/目录 | 用来回答什么问题 |
|---|---|
| manifest.json | 什么算法、训练还是评估、编码版本、依赖版本 |
| config.json | 这次实际用了什么参数 |
| status.json | 进程/出生身份、状态、最后一次进度和错误 |
| metrics/ | Episode Return、五 Agent 回报、actor/critic loss、entropy |
| frames/ | 逐步动作、观测、红队记录；部分帧含真实拓扑 |
| checkpoints/ | 按配置频率及最终 Episode 保存的权重 |
| evaluation.json | 每 seed 回报、seed 均值及样本标准差 |
| tensorboard/ | 仅当 tensorboard: true 时产生事件 |

每次运行创建独立目录，不覆盖已有实验。保存配置后再运行，文件名本身不是充分的实验说明，config.json 才是实际参数记录。

## 6. 评估一个训练模型（远程终端）

从刚完成的 run 中选择 checkpoint。下面先设置真实路径，不能直接照抄示例目录名：

```bash
cd "$HOME/cage4-research/cage4-baselines"
RUN_DIR="$HOME/cage4-experiments/REPLACE_WITH_REAL_RUN_ID"
CHECKPOINT="$RUN_DIR/checkpoints/episode_000030.pt"
test -f "$CHECKPOINT"
```

只在文件存在后运行：

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines evaluate \
  --config configs/evaluate/actor_critic.yaml \
  --checkpoint "$CHECKPOINT" --seeds 1001 1002 1003 \
  --runs "$HOME/cage4-experiments"
```

此配置默认每 seed 评估 5 Episode。评估不更新模型权重，结果写入新的独立 run。evaluation.json 中 mean 是各 seed 均值的平均，sample_std 是这些 seed 均值的样本标准差；单 seed 不提供样本标准差。

训练多 seed 时应独立训练多个模型，再做独立评估。只有一个训练模型在多个 seed 上评估，不等于已经完成多训练 seed 的统计。不要把训练曲线作为泛化成绩。

## 7. 跑 Random / Sleep 对照

仍在远程研究项目根目录：

```bash
"$HOME/cage4-research-env/bin/python" -m cage4_baselines evaluate \
  --config configs/baselines/random.yaml --seeds 1001 1002 1003 \
  --runs "$HOME/cage4-experiments"

"$HOME/cage4-research-env/bin/python" -m cage4_baselines evaluate \
  --config configs/baselines/sleep.yaml --seeds 1001 1002 1003 \
  --runs "$HOME/cage4-experiments"
```

保持红队、steps、评估 seeds 和每 seed Episode 数一致。Random/Sleep 不传 checkpoint。Team Return 是五个官方 wrapper 回报之和，可能包含重复的共享团队奖励；比较时必须保持同一语义。

## 8. 网页怎样看

- **指标**：状态、Episode、Step、团队回报、五 Agent 回报，以及真实输出的 actor loss、critic loss、entropy。Random/Sleep 无训练损失是正常情况。
- **回放**：选择 Episode、Step，读取实际保存的动作和 observation。拓扑按 snapshot_every 保存；没有拓扑的中间帧不会伪造一张网络。
- **拓扑**：链路来自实际 interface.data_links。物理链路不等于实时流量可达性，红色节点表示有红队 Session。
- **红队记录**：显示 Session、执行动作和官方 observation。invalid 表示记录到 InvalidAction；not_rejected 不等于攻击成功。
- **对比**：选择训练、模型评估及 Random/Sleep run；不同编码/步长/红队的结果不能直接解释为算法提升。
- **旧记录**：兼容显示原格式；旧数据缺少的分项 loss 无法补出来，旧 PID 身份不完整时会标注未验证。

Ground Truth 仅供研究者查看，不能作为 Blue 策略输入。内置算法只接收该 Blue 实际 observation 的编码与官方动作掩码。

## 9. 修改哪个文件

| 你想做的事 | 首先修改 |
|---|---|
| 调 seed、学习率、轮数、记录频率 | configs/ 中复制出的 YAML |
| 改 Actor-Critic 网络或损失 | algorithms/actor_critic.py |
| 增加新算法 | algorithms/ 新模块 + registry 注册 + tests/ |
| 增加超参数 | config.py 字段/范围 + YAML + 测试 |
| 改 Blue 特征 | envs/observations.py，并更新 encoder 版本与兼容说明 |
| 改场景或环境 API | envs/factory.py / envs/adapter.py |
| 改实验保存或评估统计 | experiments/runner.py / protocol/ |
| 改图表或网页布局 | 仓库 server/src/cage4_server/ |
| 改连接目标或端口 | 重新构建 Client 或其私人 settings.local |

添加算法完整契约见 [研究扩展](RESEARCH_zh.md)。不要在只读 Server 里添加“临时训练代码”。

## 10. 停止、关闭与服务维护

训练前台运行时 Ctrl+C 请求在步间安全停止；未完成 Episode 不更新权重。当前没有断点续训功能，已完成 checkpoint 可以单独评估。不要把 evaluate --checkpoint 当作 resume。

如果要让训练不随 SSH 窗口关闭，可在远程 tmux 等会话管理工具中运行（需自行安装配置），或保持远程 PyCharm 运行会话。关闭浏览器本身不会发送停止请求。

服务管理在远程固定发布目录执行，不是在本地 MacBook：

```bash
cd "$HOME/cage4-studio-releases/CAGE4_Studio_v0.3.0"
"$HOME/cage4-studio/.conda-env/bin/python" ops/service.py status
```

start/stop/install/rollback 的用法见 INSTALL_zh.md；stop 只停 dashboard，不停研究进程或旧 TensorBoard。LaunchAgent 登录后启动，不承诺开机未登录启动。回滚使用 install 输出的真实备份路径；旧安装不能删除。

本地停止客户端隧道：在 .app 所在目录执行 `bash 'CAGE4 Studio Client.app/Contents/Resources/connect.sh' stop`。这与停止远程 Server 或训练是三件不同的事。

## 11. 出现问题时

| 现象 | 首先检查 |
|---|---|
| 客户端打不开网页 | SSH 密钥/主机指纹、远程服务、本地 18501 占用与 Client 日志 |
| 网页没有新实验 | --runs 是否为 Server 读取的目录，刷新 run 列表 |
| 配置文件找不到 | 当前工作目录与 --config 相对路径 |
| Unknown algorithm / MAPPO | 名称是否注册；IPPO/MAPPO 未实现 |
| checkpoint 不兼容 | 编码版本、obs_dim、动作空间、是否使用旧 BlueFlat 权重 |
| 研究 import 失败 | PyCharm 是否使用专用远程解释器、editable 包路径是否正确 |
| 状态 interrupted | writer 已退出或 PID 被复用，查看 status.json 错误栈 |
| heartbeat 很旧但进程存在 | 研究日志/耗时仿真，网页不会擅自杀进程 |
| MPS 失败 | 切回 CPU；MPS 未做本次验收 |

## 12. 提交代码前

从仓库根目录设置 CAGE4_PYTHON 后运行 `bash ops/check.sh`，会检查仓库级和研究项目两套测试。改到环境/训练/评估时，再按 ops/validate_simulator.py 做真实 CybORG 验收。CI 只跑轻量测试，不能代替实机验证。

Git 只提交源码、YAML、测试和文档；不提交环境目录、实验记录、checkpoint、原始观测、密钥或私人连接参数。兼容性与已知未实现能力见 VALIDATION.md，不把短程 smoke test 当成科研结论。

## 0.4.0：论文级算法比较与安全预算

基础 quickstart 用于检查环境；论文比较请进入 [五算法研究说明](PAPER_BASELINES.md)。PyCharm 运行模块改为 `cage4_baselines.paper`，选择 `configs/paper/` 内相应 YAML。`safety/` 定义 cost，`configs/safety/` 保存独立校准预算；不要在 Client 或网页修改算法。网页新增业务失败率/预算曲线，回报默认按共享官方 reward 每步计一次；对比页面显式选择同一回报口径。

预算由 [官方业务权重 + 独立实测校准](SAFETY_COST_zh.md)确定，默认主阈值24.748%，并有严格/宽松档。完整训练需多训练 seed 和独立评估；验收中的短回合不能当作论文性能。升级仍备份旧服务版本和 plist，研究工程与实验目录保留；0.3 Client 可继续使用。
