# Poster 制作材料

## 1. 一句话主结论

> QLoRA efficiently adapts a 3B model to role-play dialogue and improves
> single-turn character fidelity in some settings, but prompting remains a
> strong baseline and multi-turn consistency still degrades.

中文版：

> QLoRA 能以极少可训练参数完成 3B 模型角色适配，但角色卡提示仍是强基线，
> 当前微调只呈现单轮改善趋势，多轮一致性仍明显不足。

## 2. 最终版面

Poster 使用 `80 cm × 180 cm` 竖版，两栏布局。两栏比三栏更适合窄长画布，
可以放大流程图、雷达图和正文。

### 左栏：Problem + Method + Evaluation Design

- 为什么角色扮演不仅是普通对话。
- Character Card 的实际使用方式。
- PIPPA-ShareGPT 与角色卡格式。
- Qwen2.5-3B + 4-bit QLoRA。
- 0.1193% 可训练参数，Adapter 约 14.8 MB。
- assistant-only supervision 和角色卡保留截断。
- 六个已完成评估的变量表。
- 三路系统：Base no card、Base + card、LoRA + card。
- 单轮五维和四轮五维 Judge。
- 自动指标与 Bootstrap CI。

### 右栏：Results + Takeaways

- 单轮胜率最高 55.6%，但 CI 包含 0。
- `train_6` 重复率最低 17.3%，仍高于基线 2.9%。
- 四轮综合：`train_5` LoRA 2.780 vs Base + card 3.523。
- 角色卡 Prompting 的收益大于当前 LoRA 的额外收益。
- 下一步：统一数据、清洗重复、加强多轮数据和人工盲评。

## 3. Poster 大数字

| 数字 | 含义 |
|---:|---|
| 3.09B | 基座模型参数 |
| 3.69M | 可训练 LoRA 参数 |
| 0.1193% | 可训练参数占比 |
| 14.8 MB | Adapter 权重约大小 |
| 55.6% | 最佳单轮胜率 vs Base + card |
| 17.3% | 最低 LoRA 重复率 |
| 2.780 vs 3.523 | `train_5` 多轮综合：LoRA vs Base + card |
| 101.6 min | RTX 4060 Laptop 上两轮 1024 上下文训练 |

## 4. Poster 使用图

1. **使用流程图**：Character Card -> system prompt -> chat template -> response。
2. **训练 Pipeline**：数据格式 -> assistant-only -> QLoRA -> Adapter。
3. **评估 Pipeline**：冻结测试集 -> 三路生成 -> Judge -> 配对分析。
4. **实验趋势图**：横轴 `train_1` 至 `train_6`，显示单轮胜率和重复率。
5. **train_4 单轮雷达图**：LoRA 与 Base + card 的五个维度。
6. **train_4 多轮雷达图**：突出 LoRA 仅 memory 接近，其他维度落后。
7. **角色卡贡献柱状图**：Base no card、Base + card、LoRA + card。
8. **配对效应图**：六次实验的单轮 95% CI 和多轮综合差值。

图表数据优先读取 `experiment_comparison.csv`、`paired_comparison.csv` 和
`train4_system_comparison.csv`。

## 5. 可直接放入 Poster 的结果文案

### Finding 1: Prompting is a strong baseline

In `train_4`, adding the character card improves the base model by 0.765
single-turn points and 1.270 multi-turn points.

### Finding 2: LoRA shows a single-turn trend

The best variants reach a 55.6% win rate against `Base + card`, but paired
95% confidence intervals still include zero.

### Finding 3: Multi-turn quality remains weaker

All six LoRA variants score below `Base + card` in the four-turn challenge,
especially in identity, coherence, style, and immersion.

### Finding 4: Repetition remains high

`train_6` reaches the lowest LoRA repetition rate at 17.3%, still about
6.0 times the `Base + card` rate of 2.9%.

## 6. `poster/poster.html` 已同步修正

当前 Poster 已完成以下修正：

1. 不再统一标注“single-turn, 100 samples”：`train_2` 为 70，
   `train_3/train_4` 的有效 Judge 数为 99。
2. Win rate 列使用 `↑`。
3. 删除“LoRA consistently achieves the highest Identity”：
   `train_2` 的 LoRA Identity 2.657，低于 Base + card 的 2.714。
4. 明确 18.1% 重复率仍约为 2.9% 基线的 6.3 倍。
5. 删除“1-2 epochs are not enough”的因果判断；现有证据更支持数据质量、重复模式和
   多轮监督不足等共同原因。
6. 区分“原始源数据约 16k”和“当前 processed 快照 11,515”。
7. 前四次实验硬件分别写为 RTX 4080 SUPER、RTX 5060、
   NVIDIA vGPU-32GB 和 RTX 4060 Laptop；`train_5` 为 NVIDIA vGPU-32GB。
8. 多轮结果表明确对应 `train_4`。

## 7. 最终视觉资产

- `poster/prompt-interaction-flow.png`
- `poster/training-pipeline.png`
- `poster/evaluation-pipeline.png`
- `poster/experiment-trends.svg`
- `poster/train4-single-radar.svg`
- `poster/train4-multi-radar.svg`
- `poster/character-card-impact.svg`
- `poster/paired-effects.svg`
- `poster/generate_charts.py`
- `poster/poster.html`

SVG 图表由 `python poster/generate_charts.py` 从 CSV 汇总表重新生成。
Poster 页脚记录数据日期，打印尺寸为 `800 mm × 1800 mm`。
