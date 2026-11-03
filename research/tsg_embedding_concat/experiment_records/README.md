# TSG 实验记录

| 实验 | 对照内容 | 新增正式更新 | 新增资格更新 |
|---|---|---:|---:|
| [普通 Qwen3-8B Table II](00_ordinary_qwen3_8b/) | 完整融合、拼接、数值控制及其他消融，共 12 组 | 30,720 | 36 |
| [01 编码器对照](01_encoder_pair/) | 普通 Qwen3-8B 和 Qwen3-Embedding-8B 各配动态与静态输入 | 10,240 | 12 |
| [02 常态事实对照](02_context_validation/) | Embedding 与词频 MLP 的动态／常态对照，复用 01 的动态 Embedding | 7,680 | 9 |
| [02a 单字段诊断](02a_field_diagnostic/) | 固定权重，分别错配 level、trend、weather | 0 | 0 |
| [02b 显式事实缓存](02b_typed_cache/) | 动态和常态文本编码、同事实结构化输入与缓存验收 | 0 | 0 |
| [03 显式事实对照](03_typed_factorial/) | Embedding 与结构化 MLP 各配动态与常态输入 | 10,240 | 12 |
| [04 拼接融合对照](04_concat_pair/) | 两个拼接组对照 03 的两个完整融合组 | 5,120 | 6 |
| [05 同事实同拼接](05_embedding_vs_mlp/) | 新训结构化 MLP，复用 04 的动态 Embedding | 2,560 | 3 |

Embedding 研究累计 35,840 次正式更新和 42 次资格更新。普通 8B Table II 的预算单独登记。冻结诊断的推理窗口数和缓存编码量见各轮完成记录。

各目录包含实验协议、登记摘要和完成记录。训练目录中的 `HISTORY.json` 保存候选步 TRAIN／VAL 指标，`AUDIT.json` 和 `COLD_RESTORE.json` 保存独立审计及冷重载结果。普通 8B 的 [完整融合配置](../ordinary_qwen3_8b/predictor/configs/dual.json) 和 [Table II 核验结果](../ordinary_qwen3_8b/evidence/TABLE2_CHECKED_CELLS.json) 随普通 8B 路线保存。

[来源与校验值](PROVENANCE.json)记录各文件的源位置和归档 SHA-256。带 `source_sha256` 的文件为摘要或省略了本机路径的版本。权重、缓存向量和逐窗口原始预测保存在实验归档。
