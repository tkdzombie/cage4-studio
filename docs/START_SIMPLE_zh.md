# 最简单的使用方法

在 Toolbox 连接的**远程 PyCharm**中打开原项目 `cage4-research/cage4-baselines`。

1. 打开项目根目录的 `run_baseline.py`。
2. 文件顶部选择 `ALGORITHM = "mappo"`，设置 `EPISODES = 2`、`SEED = 101`。
3. 在这个文件里右键 → **Run 'run_baseline'**。不需要填写模块名称、参数或工作目录；已有服务器 Conda Python 和独立研究 Python 都可用。断点调试选择 Debug。
4. 打开原来的 Client.app，在网页选最新实验；第一个完整回合结束后显示指标。App 只看记录，关闭它不影响训练。

可选算法：`ippo`、`mappo`、`happo`、`mappo_lagrangian`、`macpo`。程序自动读取对应 YAML；训练默认完整500配置步，不用记 cost/budget/源码缓存路径。切换算法只改第一行字符串。

以后写自己的算法：

1. 在项目根目录 `my_algorithm.py` 中编写自己的更新逻辑/网络/训练器，保留类接口。
2. 把 `run_baseline.py` 的 `ALGORITHM` 改为 `"my_algorithm"`，仍然右键运行同一个入口。
3. 示例默认沿用 MAPPO，所以可以先跑通，再逐步改 `update()`。它是扩展模板，不是第六个独立论文算法。文件注释说明了 rollout 与返回指标格式；源码 hash 会随 checkpoint 保存，避免修改算法后误加载旧权重。

正常只调整算法、轮数、seed，用启动文件顶部三项；学习率等参数放在 `configs/paper/` 对应 YAML。环境代码、业务 cost、网页服务不需要因换算法而修改。若要更改环境或 cost，按完整研究协议重新验证/校准。

现有模块 CLI 仍保留用于批量实验和独立评估，详见 PAPER_BASELINES.md。上述入口会在标准实验目录保存配置、源码来源、指标、回放和 checkpoint；并不省略实验记录或改变奖励/预算口径。
