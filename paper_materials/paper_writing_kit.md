# 论文写作材料

## 1. 推荐题目

中文：

> 基于 QLoRA 的小型语言模型角色扮演对话微调与多维评估

英文：

> QLoRA Fine-Tuning and Multi-Dimensional Evaluation for Role-Play Dialogue with a Small Language Model

## 2. 研究问题

1. LoRA 是否能在角色卡 Prompting 基线之上提升单轮角色一致性？
2. 保留角色卡的消息级截断和更长上下文是否改善角色表现？
3. 降低学习率并训练更久能否减少重复生成？
4. 单轮收益能否转化为多轮身份、记忆和连贯性收益？
5. 清洗重复数据和多轮窗口训练能否缓解多轮退化？

## 3. 可写成贡献的内容

1. 实现一套面向消费级 GPU 的 QLoRA 训练流程，包含资源发现、冒烟测试、
   时间预检、断点续训和运行清单。
2. 实现 assistant-only supervision 和保留完整角色卡的消息级截断。
3. 设计 `Base no card / Base + card / LoRA + card` 三路消融，分离 Prompting
   与微调的贡献。
4. 同时使用 PPL、生成多样性、重复率、单轮 Judge、多轮 Judge、配对 CI 和
   顺序一致性分析角色扮演质量。
5. 报告负面结果：低 PPL 不等于高角色质量，且当前 LoRA 多轮表现稳定退化。
6. 评估 assistant turn 多窗口训练方案，发现它只小幅改善 memory 维度，未改善多轮综合。

## 4. 摘要草稿

本项目研究在有限计算资源下，参数高效微调能否提升小型语言模型的角色扮演
能力。我们使用 PIPPA-ShareGPT 对 Qwen2.5-3B-Instruct 进行 4-bit QLoRA
微调，并比较角色卡保留截断、上下文长度、学习率、训练轮数、数据清洗和
assistant turn 多窗口训练。为区分角色卡提示与微调的作用，评估同时比较无角色卡
基座模型、带角色卡基座模型和带角色卡 LoRA 模型。自动指标与 MiniMax-M3 匿名
Judge 共同覆盖单轮角色身份、风格、相关性、自然度、沉浸感，以及多轮记忆和
连贯性。较优配置对带角色卡基线取得 55.6% 的单轮胜率，并将重复率从早期实验的
40.2% 降至 17.3%，但单轮配对差异的 95% 置信区间仍包含零。六个 LoRA 模型的
多轮综合分均低于带角色卡基线。结果表明，QLoRA 能有效学习角色对话分布并减少
部分元话语，但角色卡提示本身已是强基线，数据质量、重复生成和长期角色一致性
仍是主要限制。

## 5. 推荐论文结构

### 1. Introduction

- 角色扮演对身份、风格、记忆和沉浸感的要求。
- 全量微调成本与小模型/LoRA 的现实价值。
- 角色卡 Prompting 是必须比较的强基线。
- 列出研究问题和贡献。

### 2. Related Work

- LoRA 与 QLoRA。
- Persona-conditioned / role-play dialogue。
- LLM-as-a-Judge 与开放式生成评估。
- 多轮一致性和长期记忆评估。

### 3. Method

- Qwen2.5-3B-Instruct 与 4-bit QLoRA。
- PIPPA 数据、消息格式和 assistant-only supervision。
- 旧左截断与保留角色卡截断。
- 训练资源管理和可复现清单。

### 4. Experimental Setup

- 六个已完成实验的参数表。
- 三路评估系统。
- 单轮 100、四轮 20 的目标设计及实际有效样本数。
- 自动指标、Judge 维度和权重。
- 1,000 次 Bootstrap 配对 CI。

### 5. Results

- 训练效率和 Adapter 大小。
- 六次评估的 LoRA 指标表。
- LoRA vs Base + card 配对差异表。
- 单轮改善趋势、多轮退化、重复率分析。
- 角色卡本身的贡献。

### 6. Discussion

- PPL 与角色质量脱钩。
- 长上下文的潜在帮助。
- 数据重复、模板化和 speaker perspective 的影响。
- Prompting 与参数更新的不同作用。

### 7. Limitations

- 数据快照不统一，跨实验不是严格单变量。
- 当前 split 角色名重叠，需重新审计。
- Judge 单一、样本量有限且强制无平局。
- `train_2` 有效单轮样本仅 70。
- 多轮固定为四个中文 prompt，任务覆盖有限。

### 8. Conclusion

- 强调“有效适配，但未稳定超过强 Prompting 基线”。
- 明确多轮一致性和重复率是后续重点。

## 6. 推荐图表

| 编号 | 内容 | 数据来源 |
|---|---|---|
| Figure 1 | 数据到 QLoRA 再到三路评估的总流程 | `training_and_evaluation_design.md` |
| Figure 2 | LoRA 注入 q/k/v/o 投影示意 | 配置与 LoRA 公式 |
| Figure 3 | 六实验单轮差值、CI 与胜率 | `paired_comparison.csv` |
| Figure 4 | 六实验重复率与 Distinct-1 | `experiment_comparison.csv` |
| Figure 5 | `train_4` 三系统单轮五维雷达图 | `train4_system_comparison.csv` |
| Figure 6 | `train_4` Base + card vs LoRA 多轮雷达图 | `train4_system_comparison.csv` |
| Table 1 | 六个训练配置 | `training_and_evaluation_design.md` |
| Table 2 | 训练效率与硬件 | `training_comparison.csv` |
| Table 3 | 六次核心 LoRA 指标 | `experiment_comparison.csv` |
| Table 4 | 配对差异与 95% CI | `results_and_findings.md` |

## 7. 写作措辞

推荐：

> LoRA shows a single-turn improvement trend over the prompted base model,
> while the confidence interval still includes zero.

> All evaluated LoRA variants underperform the prompted base model in the
> four-turn challenge.

避免：

> LoRA significantly outperforms the base model.

> Longer context definitively causes the improvement.

> The dataset split completely removes character leakage.

## 8. 证据引用优先级

1. `summary.json` 中的同次配对差异和 CI。
2. `run_manifest.json` 中的配置、硬件、耗时和数据哈希。
3. `report.md` 和逐实验笔记中的解释与案例。
4. `experiment_comparison.csv` 用于跨实验图表，但必须附带可比性限制。
