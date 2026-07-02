# 复现性与限制审计

## 1. 训练数据快照

六次训练的 `run_manifest.json` 显示多个数据快照或同名目录快照：

| 实验 | Train SHA256 前 12 位 | Val 前 12 位 | Test 前 12 位 |
|---|---|---|---|
| `train_1` | `1d74d16726c3` | `3b9d55727382` | `9ddba961031f` |
| `train_2` | `764eed769a8d` | `4364a38c02bc` | `1f4ac7ef732d` |
| `train_3` | `1031de70542f` | `573194eced47` | `689f43b2c230` |
| `train_4` | `1d74d16726c3` | `3b9d55727382` | `9ddba961031f` |
| `train_5` | `1031de70542f` | `573194eced47` | `689f43b2c230` |
| `train_6` | `dc6e868e1a31` | `7e57a2f47fce` | `689f43b2c230` |

当前 `processed/` 为 `train_3` 快照，包含 9,211/1,152/1,152 条。
当前本地 `processed_clean/` 为 8,631/1,087/1,152 条，其 train/val hash
分别为 `dc6e868e1a31` 和 `7e57a2f47fce`，与 `train_6` 清单一致，但与
`train_5` 清单不一致。

因此：

- `train_1` 与 `train_4` 使用同一数据快照；
- `train_2` 使用独立快照；
- `train_3` 使用当前 `processed/` 磁盘快照；
- `train_5` 配置记录 `processed_clean`，但清单 hash 与当前 `processed/` 一致；
- `train_6` 使用当前 `processed_clean` 快照，并通过 assistant 窗口生成训练样本；
- `train_2` 与 `train_3` 的差异不能视为纯截断或上下文单变量消融；
- `train_3` 与 `train_4` 的差异也同时包含数据快照变化。
- `train_5` 不能作为严格清洗数据单变量实验，只能作为一次已完成的清洗目录运行清单。
- `train_6` 的改动同时包含清洗快照和窗口采样策略，不宜写成单一因素因果结论。

## 2. Split 泄漏审计

代码的设计目标是用 `GroupShuffleSplit` 按 `bot.name` 分组。但对当前
`processed/` 的实际检查得到：

| Split 对 | 完全相同角色名重叠数 |
|---|---:|
| Train / Validation | 458 |
| Train / Test | 472 |
| Validation / Test | 320 |

完整 `conversations` 内容未发现跨 split 完全重复，但角色名大量重叠。这说明
当前磁盘快照并不能证明角色级隔离已经生效，可能是旧数据沿用、生成流程未重新
执行，或分组键不足以唯一标识角色。

论文中应写成“实现了 group-aware splitting 逻辑”，不能写成“已验证完全消除
角色泄漏”。后续应：

1. 使用规范化后的 `name + description hash` 作为 group key；
2. 生成后自动断言三个 split 的 group 交集为空；
3. 将 split 统计和 group key 版本写入 dataset manifest；
4. 用固定快照重新训练关键实验。

## 3. 评估样本量

评估目标为 100 个单轮和 20 个四轮角色，但实际 Judge 结果为：

| 实验 | 单轮记录 | 有效单轮 Judge | 多轮 Judge |
|---|---:|---:|---:|
| `train_1` | 100 | 100 | 20 |
| `train_2` | 70 | 70 | 20 |
| `train_3` | 100 | 99 | 20 |
| `train_4` | 100 | 99 | 20 |
| `train_5` | 100 | 100 | 20 |
| `train_6` | 100 | 100 | 20 |

`train_2` 的 manifest 仍记录目标 `single_samples=100`，但 summary 只包含 70。
跨实验绘图必须使用有效样本数作为注释。

## 4. Judge 限制

- 仅使用 MiniMax-M3 一个 Judge，没有人工评分或第二模型交叉验证。
- 顺序一致性只抽查约 10% 样本，`train_3` 仅 50%。
- JSON schema 强制完整排序，无法表达平局，报告中的 tie rate 恒为 0。
- 单轮 CI 均包含 0，不能声称稳定显著提升。
- 四轮挑战固定为四条中文 prompt，角色原始语言和任务分布可能不匹配。

## 5. 训练环境差异

| 实验 | Python | PyTorch / CUDA | GPU |
|---|---|---|---|
| `train_1` | 3.12.3 | 2.5.1 / 12.4 | RTX 4080 SUPER |
| `train_2` | 3.11.9 | 2.11.0 / 12.8 | RTX 5060 |
| `train_3` | 3.10.20 | 2.12.0 / 13.0 | NVIDIA vGPU-32GB |
| `train_4` | 3.10.20 | 2.12.0 / 13.0 | RTX 4060 Laptop |
| `train_5` | 3.10.20 | 2.12.0 / 13.0 | NVIDIA vGPU-32GB |
| `train_6` | 3.10.20 | 2.12.0 / 13.0 | NVIDIA vGPU-32GB |

训练质量指标可比较趋势，但训练速度不能直接用于硬件效率排名。

## 6. 测试状态

当前检查结果：

- 不依赖训练框架的子集：10 tests passed。
- `ml_roleplay` Conda 环境完整测试：63 passed、4 failed、1 skipped。
- 默认系统 Python 缺少 `torch` 和 `transformers`，不能直接运行完整测试。

四个失败包括：

1. 三个测试仍引用已删除的 `configs/lora_config.yaml`，与默认推理配置失效是
   同一个问题。
2. `assistant_perplexity()` 在 context 为空、只有 reference assistant 的边界
   用例中返回 `None`。正式单轮评估会要求最后一轮 assistant 前存在 user，
   因此现有归档结果不走该路径，但函数和测试预期仍需统一。

测试覆盖的关键行为包括：

- assistant-only labels 和截断边界；
- 数据与模型资源完整性；
- best checkpoint 配置透传；
- PPL prompt mask；
- 多轮历史传递；
- 评估 resume；
- 基线复用的配置和样本一致性校验；
- Judge 重试、解析、汇总和 Bootstrap CI。

## 7. 当前实现限制

1. `scripts/inference.py` 的交互模式没有把历史传给生成函数，实际为单轮演示。
2. 默认推理配置 `configs/lora_config.yaml` 已删除，推理脚本的默认路径失效。
3. `assistant_perplexity()` 的无 prompt 边界行为与单元测试预期不一致。
4. 评估 manifest 记录数据目录字符串，但没有直接记录评估数据 SHA256。
5. 根目录 README 中部分 `output`/`outputs` 和早期配置描述存在历史差异。
6. `poster/` 当前未纳入 Git，协作时容易丢失版本。

## 8. 论文使用检查表

- [ ] 每个结果表注明有效样本数。
- [ ] 单轮结论同时给出配对差异和 95% CI。
- [ ] 多轮结论注明 20 个角色、固定四轮 prompt。
- [ ] 跨实验表注明数据快照和硬件不统一。
- [ ] 不把 PPL 降低等同于角色质量提升。
- [ ] 不宣称已彻底消除角色泄漏。
- [ ] `train_6` 写为已完成的多轮窗口负面结果，`train_5` 写为已完成但快照受限。
- [ ] Poster 与论文使用同一版 `experiment_comparison.csv`。
