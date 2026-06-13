# 论文与 Poster 材料

本目录保存适合通过 Git 同步的轻量实验材料。原始模型权重、checkpoint、
tokenizer、训练日志和逐样本评估结果仍保留在 `output/`，不纳入 Git。

## 优先使用

- `experiment_comparison.csv`：四次完整实验的核心 LoRA 指标，可直接用于表格或绘图。
- `evaluation/train_*/report.md`：人类可读的完整评估报告，含核心结果、案例和结论。
- `evaluation/train_*/summary.json`：论文统计分析的原始汇总，含均值、标准差、95% 置信区间和配对差异。
- `evaluation/train_*/manifest.json`：评估模型、样本数、随机种子、解码参数和 Judge 设置。
- `training/train_*/run_manifest.json`：训练配置、环境、数据哈希、耗时和训练指标。
- `training/train_*/benchmark.json`：50-step 预检结果和预计训练耗时。
- `notes/修改记录.md`：项目方法和实验演进的简要说明。
- `notes/第*次微调结果的记录.md`：每次实验的配置、结果与详细解释，适合整理演讲讲稿和论文分析。

## 实验状态

| 实验 | 主要变量 | 训练记录 | 完整评估 | 状态 |
|---|---|---|---|---|
| `train_1` | 512 上下文、旧左截断 | 有 | 有 | 基线 |
| `train_2` | 512 上下文、保留角色卡 | 缺少运行清单 | 有 | 截断消融 |
| `train_3` | 1024 上下文、保留角色卡 | 有 | 有 | 单轮表现较好 |
| `train_4` | 学习率 `1e-4`、2 epochs、最佳模型 | 有 | 有 | 重复率最低 |

## 引用注意

- `train_2` 实际只有 70 条单轮 Judge 结果，`train_1` 和 `train_3` 为 100 条。
  跨实验比较只能用于趋势分析，优先引用各实验内 `LoRA + card` 与
  `Base + card` 的配对差异。
- `train_3` 有 1 次 Judge API 失败，单轮 Judge 有效样本为 99 条。
- `train_2` 缺少训练日志和 `run_manifest.json`，训练耗时、loss 和数据哈希
  不应写成实测结果。
- `train_4` 有 1 次 Judge API 失败，单轮 Judge 有效样本为 99 条；其基座回答
  复用自 `train_1`，因此基座性能字段为 `N/A`。
- 根目录 `REPORT.md` 是早期草稿，配置和结果尚未同步到最新实验，不应作为最终数据来源。

## 未纳入 Git

- `adapter_model.safetensors`、optimizer 和 checkpoint：体积大，协作时应使用云盘或模型仓库。
- `single_turn_samples.jsonl`、`multi_turn_samples.jsonl`：包含完整角色文本和生成结果，
  仅在需要案例复核时单独共享。
- `excluded_samples.jsonl`：主要用于排查过滤规则，不是 Poster 或论文核心材料。
- `console.log`：详细训练日志体积较大，核心指标已保存在运行清单和实验笔记中。
