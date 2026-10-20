# CAGE4 Studio Complete v0.1.4 — 更新说明

此版本将先前 v0.1.0～v0.1.3 分散的脚本、修复和使用说明汇总为一个软件包，不需要额外叠加旧版本 ZIP、`install_fixed.sh` 或 `requirements-core-fixed.txt`。

## 合并的历史修复

1. 避免 Miniforge `.sha256` 的 404 URL，固定官方 ARM64 安装器与内置 SHA256 校验。
2. 将 `ray[rllib]==2.40.0` 调整为兼容 `gymnasium==0.28.1` 的 `ray[rllib]==2.8.1`，保留 `pettingzoo==1.24.3`。
3. 可编辑安装后以 `site-packages/cage4_repo.pth` 注册官方源码，解决 `No module named CybORG`。
4. 固定 `setuptools==80.9.0`，修复 Ray 2.8.1 导入 `pkg_resources` 问题；避免每次安装将 setuptools 自动升级到不兼容版本。
5. 已安装环境重新运行安装器时，先做 smoke test；通过则跳过重型依赖重新安装。
6. 新增 `check.sh` 与完整分步骤中文文档，并整理目录。

## 功能清单

- Mac Studio 上的 Streamlit 浏览器控制台、TensorBoard 日志监控。
- CAGE Challenge 4 环境初始化、五蓝队可执行的动作/观测接口。
- 基于实际官方 `interfaces[*].data_links` 的拓扑展示和红队 Session 标记。
- 五蓝队 on-policy Actor-Critic 研究基线、训练回报与动作历史、独立训练进程、模型 Checkpoint。
- 在本地电脑运行 `local/connect.py` 自动 SSH 端口转发并打开浏览器；PyCharm 远程解释器写代码。

## 验证范围

已完成 Python 源码语法分析、全部 `.sh` 的 `bash -n` 验证、两个拓扑与配置单元测试。用户 Mac Studio 上已实际通过 CybORG / CAGE 4 `smoke_test.py`（93 hosts / 92 interface links）。**未在生成环境中执行 Mac Studio 的 UI 与完整训练流程测试**，应在服务器上进行 2 Episodes × 10 Steps 的实验验收。
