# CAGE4 Studio Complete v0.1.4

**适用设备**：Mac Studio M3 Ultra（96GB），macOS / Apple Silicon (arm64)，Python 3.10。

**定位**：CybORG + CAGE Challenge 4 的单机可视化实验工作台（非官方）。以浏览器作为图形界面，PyCharm 负责写代码和远程调试，SSH 隧道负责安全访问。**不是**已签名的 `.app`，也不是可以替代 PyCharm 的 IDE。

## 从 GitHub 获取

本仓库是 v0.1.4 源码整理版，私有仓库需要有访问权限的 GitHub 账号。

```bash
git clone https://github.com/tkdzombie/cage4-studio.git
cd cage4-studio
chmod +x *.sh
```

全新部署执行 `bash install.sh`；已有环境执行 `bash check.sh`，通过后执行 `bash start.sh`。本机访问 http://127.0.0.1:8501，远程连接见中文连接手册。

仓库不包含虚拟环境、官方 CybORG 仓库副本、训练数据、日志或模型权重。文档中的 `YOUR_USER` 和服务器地址需替换为自己的设置。

## 已有环境升级

如果已有环境输出 `WRAPPER STEP PASS: 5 blue observations`、`SMOKE TEST PASS: 93 hosts 92 interface links`。则可复用环境，仅更新程序代码。

在 Mac Studio 上用新版 ZIP **覆盖程序代码**（不会删除现有 `.conda-env/`、`.miniforge3/`、`cage-challenge-4/` 或 `runs/`）：

```bash
mkdir -p ~/cage4-studio
unzip -o ~/Downloads/CAGE4_Studio_M3_Ultra_Complete_v0.1.4.zip -d ~/cage4-studio
cd ~/cage4-studio
chmod +x *.sh
bash check.sh
bash stop.sh
bash start.sh
```

如果 ZIP 不在 `~/Downloads/`，把上述命令改成 ZIP 实际路径。**不要在已安装成功的机器上再次执行大规模 `pip install`**。上面的 `stop.sh` 只关闭控制台和 TensorBoard，不会直接终止训练子进程。升级可视化代码前，建议先从界面停止训练或等待训练完成。

Mac Studio 本机浏览器访问 `http://127.0.0.1:8501`（TensorBoard：6006）。从其他电脑访问时必须用 SSH 隧道，见 `docs/02_PyCharm与本地连接.md`。

## 文件导览

| 位置 | 用途 |
|---|---|
| `00_先读我.txt` | 最短命令和注意事项 |
| `docs/01_服务器部署与升级.md` | 全新安装、旧版升级、验收 |
| `docs/02_PyCharm与本地连接.md` | SSH / PyCharm / 浏览器自动打开 |
| `docs/03_控制台与实验流程.md` | 配置、训练、拓扑、导出、实验科学性 |
| `docs/04_故障排查.md` | 你之前遇到的 404 / 依赖冲突 / 导入失败 / pkg_resources / 镜像慢 |
| `docs/05_文件结构及安全.md` | 各文件、数据存储和备份、安全边界 |
| `install.sh` | 全新安装；对**已成功安装**的环境优先走免重装复用路径 |
| `check.sh` | 无网络依赖的当前环境自检与官方仿真冒烟测试 |
| `start.sh` / `stop.sh` | 启动/关闭 Web UI 和 TensorBoard |
| `repair_cyborg_import.sh` | 原项目可导入路径缺失时注册 `.pth` 并做测试 |
| `requirements-core.txt` | 锁定经过冲突修复的主要兼容版本 |
| `local/connect.py` | **在本地电脑**使用的 SSH 隧道与浏览器启动器 |
| `cage4studio/ui.py` | Streamlit 控制台 |
| `cage4studio/train.py` | 五蓝队 Actor-Critic 研究基线 |
| `cage4studio/topology.py` | 真实接口链路、Red Session 可视化 |
| `smoke_test.py` | 官方接口验证 |

## 全新安装（没有任何已有环境时）

```bash
mkdir -p ~/cage4-studio
unzip -o ~/Downloads/CAGE4_Studio_M3_Ultra_Complete_v0.1.4.zip -d ~/cage4-studio
cd ~/cage4-studio
chmod +x *.sh
PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple bash install.sh
```

安装器需要网络和 Xcode Command Line Tools，不用 `sudo`。会克隆官方 CAGE Challenge 4 源码（源码和大型依赖**不包含在 ZIP 中**），并执行真实仿真测试。已知依赖修复包含：`ray[rllib]==2.8.1`、`gymnasium==0.28.1`、`setuptools==80.9.0` 及 `CybORG` 源码路径注册。

第一次实验建议 Episodes=2、Steps=10、Seed=7629、Device=cpu。成功产出 `runs/.../metrics.jsonl` 和 `tensorboard/` 后，再进行长时间训练。

## 已验证与边界

- **用户服务器已验证**：CybORG 导入、五蓝队 Wrapper 一次动作、93 主机 / 92 接口链路的仿真 smoke test。
- **本 ZIP 发布前静态验证**：源文件 Python 编译、Shell 语法、拓扑/配置单元测试、压缩包完整性。
- **仍需实际验证**：这版界面的全链路训练与 PyCharm 本地连接；具体训练效果尚未证明。软件内 Actor-Critic 是教学型独立策略梯度基线，**不是** MAPPO / PPO / 正式 CAGE 4 成绩。
- **安全**：默认只监听 `127.0.0.1`，请用 SSH 端口转发访问。不要对公网直接打开 8501 / 6006，也不要将 Ground Truth 作为模型 Observation。

官方项目：https://github.com/cage-challenge/cage-challenge-4
