# 01｜Mac Studio 上安装或升级

## 系统要求

- macOS Apple Silicon `arm64`（针对 M3 Ultra 96GB 优先测试）
- `git`、`curl`、`ssh`、Xcode Command Line Tools，联网可用（首次安装需要 GitHub）
- 初次安装预留至少约 15 GB 可用空间，以实际第三方包和实验数据为准
- SSH 使用普通登录用户，不使用 sudo

先检查：

```bash
uname -m
sw_vers
xcode-select -p
```

## A. 你当前的服务器：已装好、已通过 smoke test

你已有 Python 3.10 arm64、CybORG、Gymnasium、Ray、PyTorch，已经看到 `SMOKE TEST PASS: 93 hosts 92 interface links`。此时只升级**程序文件**：

```bash
mkdir -p ~/cage4-studio
unzip -o ~/Downloads/CAGE4_Studio_M3_Ultra_Complete_v0.1.4.zip -d ~/cage4-studio
cd ~/cage4-studio
chmod +x *.sh
bash check.sh
bash stop.sh
bash start.sh
```

`unzip -o` 覆盖同名源码文件，**不会删除** `.miniforge3`、`.conda-env`、`cage-challenge-4`、`runs`；不要先 `rm -rf ~/cage4-studio`。如果当前训练正在运行，请先在控制台点停止、等待 finished/stopped，再升级 UI。

`bash check.sh` 不会安装任何东西，会尝试官方仿真接口测试。如果 PyCharm 连接端已经建立 SSH 隧道，浏览器刷新即可。

> 已有环境不必再次执行 install.sh。即使执行，新安装器对正常现存环境会优先做 smoke test，成功则复用现有环境并跳过巨量依赖下载。

## B. 新 Mac 首次安装

把 ZIP 上传/复制到 Mac Studio Downloads，执行：

```bash
mkdir -p ~/cage4-studio
unzip -o ~/Downloads/CAGE4_Studio_M3_Ultra_Complete_v0.1.4.zip -d ~/cage4-studio
cd ~/cage4-studio
chmod +x *.sh
PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple \
PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5 \
bash install.sh
```

安装器依次处理 Miniforge ARM64 SHA256 校验、Python 3.10、官方 Git 仓库、兼容依赖、CybORG `.pth` 导入配置、实际仿真 smoke test，并在成功后启动 Streamlit 和 TensorBoard。网络速度不确定，首次依赖安装可能较久。**ZIP 不包含官方仓库和第三方 wheel**，不可离线直接安装。

若 Xcode CLT 缺少，先 `xcode-select --install`。若官方 GitHub 或镜像不可达，解决网络访问后重试，同一项目目录可以重复运行；避免清空既有目录。

## C. 固定环境与安装校验

成功标志：

```text
WRAPPER STEP PASS: 5 blue observations
SMOKE TEST PASS: 93 hosts 92 interface links
```

主机、链路数量是你此前相同 seed 对应的一次实际结果，其他生成条件下可不同，不能用固定数字判断所有场景。

检查版本和路径：

```bash
cd ~/cage4-studio
bash check.sh
cat installed-revision.txt
```

`installed-pip-freeze.txt` 记录 Python 依赖版本，`installed-revision.txt` 记录官方仓库 commit。需要可复现实验时，请把这两份文件连同训练配置一起保存。

## D. 启动、关闭、查看错误

```bash
cd ~/cage4-studio
bash start.sh
# 浏览器（Mac Studio 本机）: http://127.0.0.1:8501
# TensorBoard: http://127.0.0.1:6006

tail -n 60 dashboard.log
tail -n 60 tensorboard.log
bash stop.sh
```

`stop.sh` 不会自动终止正在运行的训练子进程。如果服务已经启动，`start.sh` 不会重复启动；需要读取更新后的 UI 文件时，先 stop 后 start。
