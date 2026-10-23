# 当前任务与参考计划

2026-10-23，已通过 Chrome 原生可访问性页面实际读取“搜集顶会论文”最新回答。

计划首阶段：CAGE4 + adityavs14/Hierarchical-MARL/h-marl-4policy；先评估作者权重，再训练子策略、Master。Ray 2.10；核对硬编码 checkpoint 路径，缩小初始采样预算。后续顺序为结构化约束 + Shield、Qwen 解析、PPO-Lagrangian/Cost Critic、语言条件代价、动态规则。OmniSafe 是约束优化参考，不替换 RLlib 主干；TTCT/SMALL 是后期代价学习参考。

当前用户要求优先：单训练 seed、短而能排除假设的实验；看清横盘/退化就停；不能将能启动/能更新称为复现成功。当前目标仅建立语义正确且实际学到提升并达到稳定平台的 baseline。暂不引入语言、旧 OT 合同、MACPO 或大规模资格验证。

源码冻结：
- Hierarchical-MARL 6fe960f5931f2d7dc5ef522f09ca4a179f377ee4
- cage-challenge-4 8c3c50ca54b176c2de199847944e8dcc035497e3


协议变更必须记录，尤其是 reward、信息边界、合法动作和 episode horizon。作者 wrapper 读取 state 中的事件/阻断/decoy 元数据；不能将其误称纯 Blue observation 输入，不能加入红队 session 真值。后续必须区分原始作者参考和任何修复协议。

当前工程状态与入口以README.md为准，本文保留研究方向来源摘要。
