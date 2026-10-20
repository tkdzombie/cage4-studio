# 文件协议 cage4.run/1

每个实验只有一个写者。研究项目创建唯一目录，观测台只读。根目录由命令行 --runs 指定，服务器的 CAGE4_RUN_ROOTS 用 macOS 的冒号分隔新旧根目录。不要把实验目录放进源码版本目录或 Git。

```
<runs>/<UTC timestamp>_<uuid>/
  manifest.json             # schema、algorithm、mode、encoder、依赖版本、seed、checkpoint SHA256
  config.json               # 实际生效的 YAML 规范化配置（JSON）
  spaces.json               # 五个 observation/action 维度
  status.json               # state/PID/identity/heartbeat/episode/step
  metrics/episode_000001/step_000019.json
  frames/episode_000001/step_000000.json
  frames/episode_000001/step_000001.json
  checkpoints/episode_000001.pt
  evaluation.json           # evaluate 才有
  tensorboard/              # tensorboard: true 才有
```

schema 字符串为 `cage4.run/1`。顶层 JSON Schema 见 protocol/schema.json，支持 manifest/status/metrics/frame。每个完整 Episode 写一条不可覆盖的 metrics：episode、step、seed、episode_seed、seed_episode、steps、team_return、agents、losses。agents 是五个 Blue 的 Episode Return；losses 是各 agent 的 actor_loss/critic_loss/entropy/loss，Random/Sleep 和评估时为空。

Team Return 定义为五个官方 wrapper reward 的和，保留原基线的语义。CAGE4 的奖励可能是共享团队值；不要将五倍共享值误解为额外性能收益。跨实验须保持相同定义。

每个实际执行 Step 写一条 frame，含动作索引/标签/掩码、Blue 原始 observation、即时 reward、done flags、Red 执行动作/observation/Session 详情。step=0 只有初始观测、拓扑和红队记录。ground-truth topology 只在第 0、1、最后一步、episode 结束及 snapshot_every 间隔保存；其他帧不伪造中间拓扑。网络边来自实际 interfaces.data_links，不等于当前流量可达性。

Red validity 的 invalid 来自官方返回的 InvalidAction；not_rejected 只说明未看到该类型，不证明攻击成功。success 应查看该 agent 的官方 observation。Blue legal_mask 是静态动作掩码，不保证动态动作成功。Session 包含 ident、hostname、username、agent、pid、parent、session_type、active。观测记录为 JSON，IP/enum/action 等非 JSON 对象转为字符串。

写入协议：同目录临时文件 → JSON（禁止 NaN/Infinity）→ flush/fsync → os.replace → 目录 fsync。只有 status.json 可更新；metric/frame 独占且不可覆盖。checkpoint 也临时写入、fsync 后 rename。临时文件以点开头，读者忽略。读者不读取任意浏览器输入路径，不跟随越出 run 的链接，单个 JSON 最大 16 MiB。原子性针对本地 APFS 等文件系统；不保证 NFS/对象存储的一致性。

状态：running/finished/stopped/error。SIGINT/SIGTERM 请求在步间停止并写 stopped；SIGKILL/断电后，通过 PID + ps 出生时间及命令比对显示 interrupted，PID 被复用也不当作原进程。无法查询身份时显示 unverified。heartbeat 只代表最后写入时刻，进程存在但 heartbeat 很旧仅提示，不擅自宣布停止。网页从不发送信号。完整状态和错误栈保存在 status.json。

encoder 为 blue-observation-hash/1：将该 agent 的实际 observation 递归展开为稳定路径/值 token，BLAKE2b 哈希到 obs_dim，计数后 tanh。无历史状态记忆，可能哈希碰撞和信息压缩，作为可运行参考而非最佳特征设计。禁止把 frames/topology 或 red 传入算法。旧 BlueFlat checkpoint 不兼容，加载时校验 encoder 和模型空间。

checkpoint 仅在可信研究命令行加载，使用 torch.load(weights_only=True)；网页从不反序列化权重。保存模型、optimizer、配置、Episode；当前只支持用于评估，尚未实现断点续训或完全 RNG 状态恢复。

评估按 seed 分组，每组 episodes 次。首个 episode_seed 等于 seed；后续由 numpy SeedSequence([seed, episode_index]) 派生。每组重设策略采样 RNG。evaluation.json 保存原始分组回报、seed 均值、这些均值的平均和样本标准差；单 seed 不提供标准差。未完成评估标记 complete=false。

## 0.4.0 向后兼容研究字段

仍使用 `cage4.run/1` 原子文件 envelope，manifest 新增 protocol=`cage4.paper-common-observation/1`、upstream commit/hash、cost_version、safety_budget_rate/level、critic_information、parameter_sharing 与 smoke 标记；config/budget/upstream-arguments 独立保存。metrics 添加 official_team_return（团队 reward 一次）、official_script_compat_return（匹配官方脚本终止步排除顺序）、safety（mean/sum、请求加权失败率、覆盖）、预算和超限量；原 team_return 保留。每个训练 agent 的 losses 保存作者字段及展示别名、模型实际权重是否改变。MACPO 上游错误的 cost_loss 诊断被省略，另写更新后 cost critic MSE。frames 新增 safety，每个实际步写入一次。详情见 PAPER_BASELINES.md / SAFETY_COST_zh.md。旧 Server 能读取 envelope，但0.4 UI 才区分这些回报口径并展示 cost。
