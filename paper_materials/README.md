# 论文与 Poster 材料索引

本目录集中保存项目实现、训练设计、评估设计、实验结果和写作素材。正文写作时，
优先引用这里的归档数据，不直接从根目录早期草稿或 Poster 页面抄数。

## 推荐阅读顺序

1. `project_implementation.md`：项目目标、系统架构、模块职责和已实现功能。
2. `training_and_evaluation_design.md`：数据处理、QLoRA 训练和三路评估方法。
3. `results_and_findings.md`：六轮已评估实验的结果、结论和待验证实验。
4. `paper_writing_kit.md`：论文题目、研究问题、贡献、章节结构和图表建议。
5. `poster_kit.md`：Poster 文案结构、核心数字、图表和现有页面勘误。
6. `reproducibility_audit.md`：数据版本、实验可比性、测试状态和已知限制。

## 原始证据

- `experiment_comparison.csv`：六次完整评估的核心 LoRA 指标。
- `training_comparison.csv`：训练参数、硬件、耗时和数据版本对比。
- `paired_comparison.csv`：LoRA 对 Prompting 基线的配对差异、CI 和可靠性。
- `train5_train6_comparison.csv`：清洗数据后缀训练与 assistant 窗口训练的直接对照。
- `train4_system_comparison.csv`：`train_4` 三系统柱状图和雷达图数据。
- `evaluation/train_*/report.md`：自动生成的人类可读评估报告。
- `evaluation/train_*/summary.json`：均值、标准差、Bootstrap 95% CI、配对差异。
- `evaluation/train_*/manifest.json`：模型、样本数、seed、生成和 Judge 设置。
- `training/train_*/run_manifest.json`：训练配置、环境、数据哈希、耗时和指标。
- `training/train_*/benchmark.json`：50-step 时间预检结果；`train_2` 无此归档。
- `notes/第*次微调结果的记录.md`：逐实验解释和代表案例。
- `notes/修改记录.md`：项目方法与工程流程的演进。

原始大文件仍保存在 `output/`，包括 checkpoint、LoRA 权重、训练日志和逐样本
评估 JSONL。它们不适合直接提交 Git，但在复核案例和重算指标时必须保留。

## 实验状态

| 实验 | 主要变量 | 训练记录 | 完整评估 | 状态 |
|---|---|---|---|---|
| `train_1` | 512 上下文、旧 token 左截断 | 有 | 有 | 初始基线 |
| `train_2` | 512 上下文、保留角色卡 | 有 | 有 | 截断消融 |
| `train_3` | 1024 上下文、保留角色卡 | 有 | 有 | 长上下文趋势 |
| `train_4` | `lr=1e-4`、2 epochs、最佳 checkpoint | 有 | 有 | 低学习率对照 |
| `train_5` | 清洗数据运行清单 | 有 | 有 | 已完成，快照需谨慎解释 |
| `train_6` | 清洗数据、多轮 assistant 窗口 | 有 | 有 | 已完成，未改善多轮综合 |

## 引用规则

- 最可靠的结论是同一评估内 `LoRA + card` 与 `Base + card` 的配对差异。
- `train_2` 只有 70 条有效单轮 Judge 结果，`train_3/train_4` 各有 99 条，
  不能把所有实验写成完全等样本量比较。
- 六次训练使用了多个数据快照或同名目录快照，跨实验差异不能只归因于单个超参数。
- `train_4` 的两组基座回答复用自 `train_1`，基座性能字段为 `N/A`。
- 根目录 `REPORT.md` 是较早论文草稿，可复用叙事结构，但数字应以本目录为准。
- `poster/poster.html` 是未跟踪的工作文件，其中部分表述需要按 `poster_kit.md`
  修正后再用于最终 Poster。

## 当前最稳妥的总述

QLoRA 明显降低了角色回复的困惑度，并在较长上下文配置下呈现单轮角色一致性
改善趋势；但六个 LoRA 模型的多轮综合表现均低于 `Base + card`，且重复率仍
明显更高。角色卡提示本身是强基线，当前证据不支持“LoRA 已稳定超过提示基线”。
