# 04｜安装历史问题与故障处理

## 1. `zsh: permission denied: ./install.sh`

```bash
cd ~/cage4-studio
chmod +x *.sh
bash install.sh
```

`bash install.sh` 可以避免脚本没有执行位的问题。

## 2. Miniforge 下载 404

旧版错误 URL 指向不存在的 `...MacOSX-arm64.sh.sha256` release asset。v0.1.4 已使用固定版本的官方 release shell 安装器并核对内置 SHA256 值，不会额外去下载无效的 `.sha256` URL。旧版脚本请直接升级到此版。

## 3. `gymnasium==0.28.1` 和 `ray[rllib]==2.40.0` 互斥

新版配置固定 `ray[rllib]==2.8.1` + `gymnasium==0.28.1`，并把旧 RLlib 相关依赖版本保守锁定。不要自己把 Ray 升级回 2.40.0。如需验证：

```bash
cd ~/cage4-studio
grep -E 'gymnasium|ray\[rllib\]|setuptools' requirements-core.txt
```

## 4. `Successfully installed CybORG-4.0` 却 `No module named 'CybORG'`

旧版可编辑安装没有自动导入源码。v0.1.4 安装器已经在私有 Python 环境 site-packages 写入 `.pth` 指向 `cage-challenge-4`。现有环境的轻量修复：

```bash
cd ~/cage4-studio
bash repair_cyborg_import.sh
```

这会运行真实接口检查，不重装大型依赖。

## 5. `No module named 'pkg_resources'`（Ray/pygame）

你已经验证 pinning setuptools 可恢复旧 Ray 依赖的导入。此版已修复安装脚本中 `pip install --upgrade setuptools` 导致的版本回退问题，固定 `setuptools==80.9.0`。必要时：

```bash
cd ~/cage4-studio
./.conda-env/bin/python -m pip install 'setuptools==80.9.0' \
  -i https://mirrors.cloud.tencent.com/pypi/simple
bash check.sh
```

`pkg_resources is deprecated` 和旧 `gym` 已不维护等警告，与你通过 smoke test 时看到的一样；它们不是此次测试失败的原因。

## 6. PyPI 下载几百 MB 包很慢

如果处于**第一次安装**，可显式设置镜像，仅影响 Python pip：

```bash
cd ~/cage4-studio
PIP_INDEX_URL=https://mirrors.cloud.tencent.com/pypi/simple \
PIP_DEFAULT_TIMEOUT=120 PIP_RETRIES=5 bash install.sh
```

Miniforge/Conda 和官方仓库分别走 conda-forge/GitHub，不受 pip 镜像影响。更换镜像无法保证绝对加速；已经 `SMOKE TEST PASS` 的环境没有必要重新安装。

## 7. `SMOKE TEST PASS` 了但网页打不开

```bash
cd ~/cage4-studio
bash start.sh
tail -n 60 dashboard.log
tail -n 60 tensorboard.log
```

Mac Studio **本机浏览器**打开 `http://127.0.0.1:8501`。另一台电脑不能把这个地址当成服务器地址直接访问，必须使用 `local/connect.py` 或 `ssh -L` 转发。若已有旧界面进程，运行 `bash stop.sh && bash start.sh`。检查端口：

```bash
lsof -nP -iTCP:8501 -sTCP:LISTEN
lsof -nP -iTCP:6006 -sTCP:LISTEN
```

不要因为端口冲突就直接用 `sudo`，检查是哪一个进程占用。

## 8. 网页可打开但训练错误

找到最新 Run，检查其 `train.log` / `status.json`：

```bash
cd ~/cage4-studio
ls -lt runs | head
tail -n 80 runs/某个Run目录/train.log
bash check.sh
```

`bash check.sh` 只验证仿真与 Blue wrapper 一步动作，**不等同于完整训练成功**。训练验证需要在 UI 中启动一个 2×10 的微型实验，确认指标文件、checkpoint 和 TensorBoard 均被写入。

## 9. PyCharm 本地启动器连接失败

本地先运行 `ssh YOUR_USER@你的服务器IP`，确认账号、IP、私钥和系统设置中的远程登录。如果启动器提示 BatchMode 权限失败，请先配置公钥免密；它不会交互式请求密码。确保本地的 8501/6006 未被其他服务占用，或用 `--local-port 8502 --tensorboard-port 6007`。
