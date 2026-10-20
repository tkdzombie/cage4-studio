# Python 依赖与运行环境

这里放解释器与依赖说明；仿真接口代码在 ../src/cage4_baselines/envs，实验 YAML 在 ../configs。三者不能混为一个“环境文件”。

| 内容 | 放在哪里 | 是否进 Git |
|---|---|---|
| Python 包的直接依赖、console command | ../pyproject.toml | 是 |
| 已验证的原 Server 关键依赖版本 | requirements-server-reference.txt | 是，仅参考，不是完整 lock |
| 当前研究解释器的实际依赖 | 每个 run 的 manifest.json / 本机 pip freeze | 实验记录不进 Git |
| 实际 Python 虚拟环境 | 远程用户 ~/cage4-research-env | 否 |
| CybORG 第三方源码 | 现有安装的 cage-challenge-4，固定上游 commit | 否，不复制进本项目 |

研究环境使用 Python 3.10，借助 --system-site-packages 复用已安装大依赖。包安装与解释器选择见仓库 docs/INSTALL_zh.md。不能直接删除旧 Server 的 .conda-env，它提供研究环境继承的依赖。

requirements-server-reference.txt 记录原环境 setuptools 80.9.0；当前独立研究环境的 setuptools 为 79.0.1，两者均经过实机验证。不要为形式上对齐参考文件而重装工作的环境。实际版本随 run 一起记录。空白机器的依赖安装与 MPS 支持仍需独立验证。

依赖只在 pyproject.toml 声明，并在本目录记录已验证版本；根目录的历史约束与转发副本已删除。新增依赖验证后同步更新本目录说明。
