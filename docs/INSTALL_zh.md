# 安装、迁移与回滚

## 两台 Mac 的职责

本地 MacBook Air：浏览器、Client.app、JetBrains Toolbox 的远程开发入口。
远程 Mac Studio：CybORG、Python 研究解释器、项目源码、实验记录和 launchd Server。
所有 train/evaluate/service 命令都在 **Mac Studio 远程终端** 执行。不要在本地误装整个模拟器。

## 从下载 ZIP 升级（远程 Mac Studio 终端）

把源码 ZIP 上传到 Mac Studio Downloads。在本地先对比同一批下载中的 SHA256SUMS.txt。包只有一层 `CAGE4_Studio_v0.3.0/`，无中文文件名、无 __MACOSX。文档内容为 UTF-8 中文。

```bash
mkdir -p $HOME/cage4-studio-releases
test ! -e $HOME/cage4-studio-releases/CAGE4_Studio_v0.3.0
unzip -q $HOME/Downloads/CAGE4_Studio_v0.3.0_Source.zip \
  -d $HOME/cage4-studio-releases
cd $HOME/cage4-studio-releases/CAGE4_Studio_v0.3.0
bash ops/check.sh
$HOME/cage4-studio/.conda-env/bin/python ops/service.py install
```

逐行执行。若 `test` 失败，说明版本目录已存在，不要继续解压覆盖；使用已有目录或另外命名新版本。不要解压到旧安装内，也不要把整个 `CAGE4_Studio_v0.3.0` 再复制到自身下一级。

service install 先检查现有 Python/import，备份原 dashboard plist 到 `~/cage4-studio-backups/<时间>/`，再切换原 com.cage4studio.dashboard。新服务仍只监听 127.0.0.1:8501，读取 ~/cage4-experiments 和旧 ~/cage4-studio/runs。健康检查失败会自动恢复原 plist 与加载状态。输出的 Backup 路径须保存。

原安装完整保留，原环境、旧数据和 TensorBoard 不改。更新前不要删除旧目录！独立研究环境通过 system-site-packages 复用其大依赖，仍依赖旧环境存在。若要独立搬迁，需重新安装依赖并验证。

## 独立研究解释器（远程终端，首次配置）

本次部署会建立 `~/cage4-research-env`，若已存在不要重建覆盖。手动安装新机器时：

```bash
$HOME/cage4-studio/.conda-env/bin/python -m venv \
  --system-site-packages $HOME/cage4-research-env
$HOME/cage4-research-env/bin/python -m pip install \
  --no-deps --no-build-isolation -e protocol -e cage4-baselines
```

为了长期研究，建议独立克隆工作副本，而不是修改 Server 固定版本目录。本次交付已完成以下步骤；新机器首次配置时执行：

```bash
git clone https://github.com/tkdzombie/cage4-studio.git "$HOME/cage4-research"
cd "$HOME/cage4-research"
"$HOME/cage4-research-env/bin/python" -m pip install \
  --no-deps --no-build-isolation -e protocol -e cage4-baselines
```

研究解释器的 editable 包最终指向工作副本；Server 的 PYTHONPATH 固定指向发布目录，不随研究项目修改。只安装本地包，不下载 PyTorch/Ray，不改变旧解释器。PyCharm 打开 `$HOME/cage4-research/cage4-baselines`，选这个新解释器。命令行便捷脚本默认用旧解释器和显式 PYTHONPATH；如需统一使用新解释器，在本次终端设置：

```bash
export CAGE4_PYTHON=$HOME/cage4-research-env/bin/python
```

## 服务操作（远程终端）

```bash
$HOME/cage4-studio/.conda-env/bin/python ops/service.py status
$HOME/cage4-studio/.conda-env/bin/python ops/service.py stop
$HOME/cage4-studio/.conda-env/bin/python ops/service.py start
```

这是当前用户的 LaunchAgent：登录该 macOS 用户后启动，不承诺开机未登录时启动。SSH 下优先选已有 gui/UID，只有不存在才尝试 user/UID；不能把它说成系统级 LaunchDaemon。未登录或域不存在时须在 Mac Studio 登录。stop 只卸载 dashboard，不停止训练和 TensorBoard。

现有 TensorBoard 仍由旧 `bash ~/cage4-studio/service.sh status` 查询；不要在新 dashboard 启用后再执行旧 service.sh install/restart，它会重新指向旧网页。若新实验需要 TensorBoard，可另开远程终端：

```bash
$HOME/cage4-studio/.conda-env/bin/python -m tensorboard.main \
  --logdir $HOME/cage4-experiments --host 127.0.0.1 --port 16006
```

这是可选操作，不能和现有 6006 服务混淆；若使用远程 16006，需手动建立对应 SSH 16006 转发。默认客户端可转发的是远程 6006。首次交付保留旧 TensorBoard 的原 logdir。

回滚（用 install 输出的真实 Backup 路径替换示例）：

```bash
$HOME/cage4-studio/.conda-env/bin/python ops/service.py rollback \
  --backup $HOME/cage4-studio-backups/<时间>
```

回滚只还原服务定义/加载状态。新旧实验均保留，源码未覆盖。端口被手动启动的非 launchd 进程占用时，安装器会报错；先核实 `lsof -nP -iTCP:8501 -sTCP:LISTEN` 的真实进程，不使用 pkill 清场。

## 本地客户端（本地 MacBook Air 终端）

在源码包内 `bash client/build_mac_app.sh <输出目录> <主机> <远程用户> [本地端口]` 可重建原生 .app。这三个连接参数仅写入生成的 .app/Contents/Resources/settings.local，不写入公共源码。也可直接使用独立 Client ZIP。它只用系统 ssh/curl/open，严格校验主机密钥，使用用户已有 ~/.ssh 配置和密钥，不内嵌私钥/密码；ControlMaster socket 位于用户私有 Application Support 目录。

通用源码默认使用 SSH 别名 cage4-studio 和当前用户名；请显式传入实际主机和远程用户，或构建时配置。私人交付的 Client.app 可单独预置连接参数。服务器必须先运行。默认本地端口为 18501，保留旧客户端占用的 8501；若 18501 已被其他程序占用，新建隧道会失败，不会借用未知监听端口。日志在 `~/Library/Logs/CAGE4-Studio-Client.log`。status/stop 见 README。自建 SSH 别名可以在源码 connect.sh 的参数传入；自定义端口/身份文件放进现有 ~/.ssh/config。

## 从空白环境重建

这次交付针对已运行的 Mac Studio，不带 Conda、PyTorch 或 CybORG 二进制，不自动下载它们。空白机器需要 Python 3.10、上游固定 commit、macOS arm64 对应 wheel。`requirements-legacy.txt` 是原始安装约束，仅供溯源；与实际已运行环境不完全一致，不要为了“对齐”而直接重装。`requirements-tested.txt` 是本次关键版本快照，并非完整可重现 lock。

如必须重建，在新环境中按关键版本安装，再 clone 上游 `https://github.com/cage-challenge/cage-challenge-4.git`、checkout `8c3c50ca54b176c2de199847944e8dcc035497e3`，使用 `pip install --no-deps -e <源码目录>`。若仍无法 import，显式将源码目录加入该新解释器的 .pth（不要改其他环境）。接着运行 ops/check.sh 与 ops/validate_simulator.py。不要使用已失效的 Miniforge checksum 下载 URL，不把 Ray 2.40 与 gymnasium 0.28.1 混装。空白安装流程未做第二台机器验收，不能视为一键支持所有平台。
