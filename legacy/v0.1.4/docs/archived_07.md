# 05｜代码结构、数据管理与安全

## 文件结构

```text
cage4-studio/
├── 00_先读我.txt
├── README_中文.md
├── docs/                  服务器 / PyCharm / 实验流程 / 故障排查 / 结构
├── install.sh             全新安装+已有环境快速复用
├── check.sh               诊断与仿真接口测试（不会安装包）
├── start.sh               Streamlit + TensorBoard，仅监听本机
├── stop.sh                关闭控制台，不强制停止训练
├── repair_cyborg_import.sh
├── requirements-core.txt
├── smoke_test.py
├── local/
│   └── connect.py        在本地电脑运行，自动 SSH 隧道 + 打开浏览器
├── cage4studio/
│   ├── engine.py          CAGE 4 仿真环境初始化
│   ├── topology.py        主机、Session、真实链路、Graphviz
│   ├── train.py           五蓝队独立 Actor-Critic 研究基线
│   ├── ui.py              Streamlit Web 控制台
│   └── runs.py            实验目录、日志状态、原子写入
├── tests/test_kit.py      不依赖模拟环境的拓扑/配置测试
├── .miniforge3/           首次安装后生成，不包含于 ZIP
├── .conda-env/            首次安装后生成，不包含于 ZIP
├── cage-challenge-4/      首次安装时官方 Git 克隆，不包含于 ZIP
└── runs/                  实验记录及 Checkpoint，不包含于 ZIP
```

## 有关已经安装的用户

覆盖 ZIP 中同名的程序文件不会删除你已有的训练记录，但请保留整个 `runs/` 目录。不要误执行 `rm -rf ~/cage4-studio`、`conda remove` 或强制升级所有包。项目若迁移到其他路径，`CybORG` 的 `.pth` 可能需要重新注册，使用 `bash repair_cyborg_import.sh`。

## 安全边界

控制台设计为**单用户研究工作台**，不是支持多租户公开访问的 SaaS。`start.sh` 默认只监听 `127.0.0.1:8501`、`127.0.0.1:6006`，只应通过 SSH 端口转发进入，不开放公网访问。在有别的用户可以登录这台机器的情况下，也需要 Linux/macOS 文件权限与 SSH 层面的访问控制。

研究者 UI 读取的真实网络状态是仿真 Ground Truth，请不要把它加入训练特征，以免泄漏导致评估无效。

## 版本锁定与验证

依赖关键版本：Python 3.10、Gymnasium 0.28.1、Ray/RLlib 2.8.1、Setuptools 80.9.0、PyTorch 2.2.0。依赖安装采用有界或固定的版本要求，但其他传递依赖可能因安装时间不同而变化；实际可复现的记录以 `installed-pip-freeze.txt` 和 `installed-revision.txt` 为准。

已在当前会话中使用你提供的终端日志确认过 smoke test，但 ZIP 在这边环境进行的是静态语法与单元测试。完整训练任务的运行情况和 UI 状态，需要你在 Mac Studio 实际启动后确认。
