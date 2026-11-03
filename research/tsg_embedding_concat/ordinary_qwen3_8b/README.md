# TSG 普通 Qwen3-8B 路线

共享预测器覆盖负荷、风电、光伏和净负荷。冻结的普通 Qwen3-8B 对每个预测窗口的文本做编码，以非填充 token 的末层状态均值生成 4096 维向量。完整双门控配置把文本向量、数值历史摘要和可用性标记输入拼接分支，同时保留语义与数值投影的逐元素乘积分支。融合向量控制各层低秩残差的成分和修正幅度。

- [当前稿件源码](manuscript/paper.tex)及参考文献、文档类和插图是归档时的同字节快照。稿件中 Table I、SMARD 的主模型结果和 CAISO 标签预算实验采用各自的 Qwen3-0.6B 字段条件化配方，按各节协议归属。与原 Table II 实验登记绑定的稿件另存为 [登记时稿件快照](evidence/paper_source_snapshot.tex)。
- [完整融合配置](predictor/configs/dual.json)与预测器、编码、训练及审计入口随档保存。完整复训依赖原四任务数据包、数据协议、普通 Qwen3-8B 权重及实验登记所绑定的语义缓存。
- [已核验的 Table II 消融](evidence/TABLE2_CHECKED_CELLS.json)包含 12 个训练组。各组按完整四任务 VAL 资产宏 NMAE 均值从每 128 步的候选中选择一个 checkpoint，同分取最早。种子为 2027，每组 2560 次正式更新及 3 次资格更新。

| Table II 配置 | 四任务平均 VAL NMAE |
|---|---:|
| 普通 Qwen3-8B 完整融合（dual） | 0.355408 |
| 数值状态控制（numeric_state） | 0.348082 |
| 普通 Qwen3-8B 拼接控制（concat_only） | 0.342964 |

Table II 对应普通 Qwen3-8B 的原事实文本与全文均值读出。Embedding 路线采用显式运行事实与末 token 读出。文件来源和归档 SHA-256 见 [PROVENANCE.json](PROVENANCE.json)。含 `source_sha256` 的条目同时记录实验原件的校验值。
