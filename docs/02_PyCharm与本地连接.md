# 02｜PyCharm、本地自动打开可视化控制台

## A. 在 Mac Studio 开启 SSH

macOS 设置 → 通用 → 共享 → **远程登录**，允许你的 SSH 用户登录。确保本地和 Mac Studio 网络互通；若跨公网，建议使用受控 VPN，不要直接暴露 Streamlit/TensorBoard 端口。

从**本地电脑终端**先验证：

```bash
ssh YOUR_USER@你的MacStudioIP
```

Mac Studio 的 IP 可以在“系统设置 → 网络”查看，或在服务器上 `ipconfig getifaddr en0`（实际网卡可能不是 en0）。建议配置 SSH 密钥登录。

## B. PyCharm 配远程解释器（用于写代码）

使用支持 SSH 解释器的 PyCharm 版本。在 Settings / Preferences → Project → Python Interpreter → Add Interpreter → On SSH 中填写远程机器、SSH 账号，并选择服务器解释器：

```text
/Users/YOUR_USER/cage4-studio/.conda-env/bin/python
```

将工作目录映射为：

```text
/Users/YOUR_USER/cage4-studio
```

如果你的服务器登录用户不同，请替换 home 路径。PyCharm 版本不同，入口菜单可能稍有不同。请注意：连接 SSH 解释器 **不保证** 自动弹出可视化窗口。

## C. 在 PyCharm 点击运行，自动打开本地浏览器（推荐）

将压缩包中的 `local/connect.py` 复制到本地电脑项目中（不需要下载 Python 依赖，只需 Python 3 与系统 ssh）。在 PyCharm 添加一个新的 **Python Run Configuration，使用本地 Python 解释器**：

- Script path：本地项目内的 `local/connect.py`
- Parameters：`--host 你的MacStudioIP --user YOUR_USER`
- Environment：本地普通 Python 环境（不是远程 SSH 解释器）

点击 Run：启动器会在 Mac Studio 执行 `./start.sh`，建立 8501/6006 SSH 转发，并使用本地默认浏览器打开 `http://127.0.0.1:8501`。运行窗口需要保持开启，关闭/按 Ctrl+C 则断开隧道，**不会自动停掉服务器端训练**。

也可在本地命令行直接执行：

```bash
python3 local/connect.py --host 你的MacStudioIP --user YOUR_USER
```

使用不同端口或特殊 SSH 私钥：

```bash
python3 local/connect.py --host 你的MacStudioIP --user YOUR_USER \
  --port 22 --identity-file ~/.ssh/id_ed25519 \
  --local-port 8502 --tensorboard-port 6007
```

这时本地浏览器将是 `http://127.0.0.1:8502`。如果 SSH 提示 BatchMode 认证失败，先在终端使用同一私钥手动 SSH 登录确认密钥已配置。

## D. 不使用启动器的手动连接方式

在服务器端确认 `bash start.sh` 已启动，接着本地终端执行：

```bash
ssh -N \
  -L 8501:127.0.0.1:8501 \
  -L 6006:127.0.0.1:6006 \
  YOUR_USER@你的MacStudioIP
```

本地浏览器打开：

- CAGE4 Studio: `http://127.0.0.1:8501`
- TensorBoard: `http://127.0.0.1:6006`

连接中断时重新运行同一个命令。不需要对外开放远程服务器的 8501 或 6006。
