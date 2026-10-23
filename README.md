# CAGE4 算法实验项目

服务器项目：`/Users/tiankedi/cage4-research`。PyCharm打开该目录，解释器选择 `.venv/bin/python`。

- `algorithms/my_algorithm.py`：自己的 PPO loss / 更新扩展；原样等价于PPO。
- `experiments/run_baseline.py`：选择ppo或my_algorithm、author或scratch初始化、唯一运行名及预算，右键Run。
- `train.py`：公共采样、训练与模型保存。
- `primitive_env.py`：固定CAGE4环境适配。
- `compare_flat.py`：完整模型恢复及共同场景评估。
- `models/ppo-author/`、`models/ppo-scratch/`：服务器保留的两份可用baseline末尾模型及其配置，不提交Git。
- `runs/`：今后新实验自动生成的输出目录，不提交Git；历史运行结果已清除。
- `dashboard.py`、`ops/dashboard.sh`：查看今后的实验曲线；`bash ops/dashboard.sh start`手动启动看板。
- `client/`：本机CAGE4客户端源码；已安装app可直接使用。

[算法修改与对比方法](docs/ALGORITHM_GUIDE_zh.md)。环境沿用作者state-derived事件/block/decoy观测，包含流量控制，官方原始团队奖励；有限回合500配置步/499实际transition。对比固定初始化、环境协议、观测权限、动作集合与交互预算。

服务器已备好独立环境、官方CAGE4与作者Hierarchical-MARL上游，不要删除 `.venv-independent`、`cage-challenge-4`、`Hierarchical-MARL`。基础runtime位于项目外的 `/Users/tiankedi/cage4-runtime-independent`。版本快照为 `runtime-freeze.txt`，核心依赖为 `requirements.txt`。

测试、诊断、旧平台、归档、历史结果、图表和重复模型包已从当前项目清除。新算法checkpoint依赖保存时的Python模块，保留算法类的导入路径。

上游工作目录也已裁剪：删除未使用的h-marl-3policy/h-marl-4policy、官方测试和说明文档，保留平坦策略运行代码、作者初始化权重和模拟器运行包。上游Git记录保留，裁剪不修改运行代码。
官方模拟器启动依赖 `CybORG/Tests/__init__.py` 和 `CybORG/Tests/utils.py`，这两个运行依赖保留，其余测试用例已删除。
