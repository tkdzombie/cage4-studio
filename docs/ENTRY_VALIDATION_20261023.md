# 算法开发入口验收（2026-10-23）

已在现有CAGE4独立解释器验证；本轮只整理开发入口，没有追加baseline长训练或改环境协议。

- PyCharm启动配置三项测试通过：author/scratch初始化、critic单位转换、不认识的初始化拒绝。
- MyAlgorithmPolicy模板：workers0、499步、1次PPO更新、1 SGD epoch，预算结束正常保存initial/best/final。训练回报−706，仅工程验收。
- 五个自定义Policy通过真实Policy.from_checkpoint恢复，在54993场景完成499步，回报−2138；不是新算法性能证据，模板仍等价PPO。
- 当前PrimitiveEnv完整499步核对通过，Blue0–3观测76/动作82，HQ观测226/动作242，mask、奖励加总、terminal/truncated与已知隐藏信息辅助函数禁用检查通过。
- 原主baseline末尾模型再次恢复，在54994场景完成499步，回报−189，reward分项−131−58，模型未更新。这个场景仅作重载检查，未追加到原四场景均值。
- 新CAGE4客户端编译、签名核验、SSH隧道和看板health通过；原CAGE3客户端可恢复归档，旧8502看板LaunchAgent已停用并备份。

完整新入口烟测checkpoint和日志位于服务器 `maintenance-backups/entry-smoke-20261023/`；不混入主学习曲线。现成baseline的源码快照仍在原run/source，根目录runner的新结构仅影响今后的run。
