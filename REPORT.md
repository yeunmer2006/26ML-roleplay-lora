# Role-Play Dialogue Generation via LoRA Fine-Tuning of a Small Language Model

**Authors:** Liu Yihan, Liao Xucheng, Long Hongtan

## Abstract

This project studies whether parameter-efficient fine-tuning can improve the
role-play ability of a small language model under limited computing resources.
We fine-tune Qwen2.5-3B-Instruct on PIPPA-ShareGPT conversations using 4-bit
QLoRA and evaluate four controlled experiments involving truncation strategy,
context length, learning rate, and training epochs. To separate the effects of
prompting and fine-tuning, we compare the base model without a character card,
the base model with a character card, and the LoRA model with the same card.
Evaluation combines automatic metrics with an anonymous MiniMax-M3 judge for
single-turn and multi-turn dialogue.

The best LoRA models obtain a 55.6% single-turn win rate against the prompted
base model and reduce refusal behavior. However, their single-turn improvement
is not statistically conclusive, and all four LoRA models perform worse than
the prompted base model in multi-turn evaluation. Lowering the learning rate
and training for two epochs reduces repetition from 23.9% to 18.1%, but does
not resolve multi-turn degradation. These results show that LoRA successfully
learns the role-play data distribution, while data quality, repetition, and
long-term character consistency remain the main limitations.

**Keywords:** LoRA, QLoRA, Role-Play Dialogue, Character Consistency,
Parameter-Efficient Fine-Tuning, LLM-as-Judge

## 1. Introduction

Role-play dialogue requires a model to follow a character description, maintain
the character's identity and speaking style, respond naturally to the user, and
remain consistent over multiple turns. Prompting a general instruction model
with a character card is a strong baseline, but it may still produce generic
assistant language, safety-related meta commentary, or out-of-character
responses.

This project investigates whether LoRA can improve role-play behavior without
full-model fine-tuning. The work focuses on three questions:

1. Does LoRA improve role identity beyond supplying the base model with a
   character card?
2. How do character-card-preserving truncation and longer context affect the
   result?
3. Can a lower learning rate and longer training reduce repetitive generation?

### 1.1 Contributions

1. We implement an end-to-end QLoRA pipeline covering data preparation,
   training, inference, checkpoint recovery, and reproducible experiment
   manifests.
2. We introduce group-aware dataset splitting, assistant-only supervision, and
   a truncation strategy that preserves the complete character card.
3. We build a three-system evaluation framework with automatic metrics,
   single-turn and four-turn LLM judging, paired confidence intervals, resumable
   execution, and reusable baseline outputs.
4. We report four controlled experiments and analyze both improvements and
   failure modes instead of relying on perplexity alone.

## 2. Related Work

### 2.1 Parameter-Efficient Fine-Tuning

LoRA freezes the pretrained model and represents each trainable update with two
low-rank matrices. QLoRA further reduces memory use by loading the base model in
4-bit precision while training LoRA parameters in higher precision. These
methods make adaptation of multi-billion-parameter models feasible on
consumer-grade hardware.

### 2.2 Persona and Role-Play Dialogue

Persona-conditioned dialogue commonly supplies a profile or character
description through a prompt. Role-play adds stricter requirements: responses
must preserve identity, style, relationships, and narrative continuity. A low
language-modeling loss is therefore insufficient to establish role-play
quality.

### 2.3 LLM-Based Evaluation

Reference-based metrics are poorly suited to open-ended dialogue because many
different responses may be valid. We therefore combine local automatic metrics
with an anonymous LLM judge and report paired comparisons against the same
prompted base-model responses.

## 3. Methodology

### 3.1 Base Model and QLoRA

The base model is Qwen2.5-3B-Instruct. It is loaded with NF4 4-bit
quantization, double quantization, and FP16 computation. Gradient checkpointing
is enabled and the KV cache is disabled during training.

For all four experiments, LoRA is applied to the attention projections:

```yaml
lora:
  r: 8
  lora_alpha: 16
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  lora_dropout: 0.05
  bias: none
```

This configuration trains 3,686,400 parameters, approximately 0.1193% of the
3.09B-parameter model.

Given a pretrained weight matrix \(W_0\), LoRA represents the update as:

\[
W = W_0 + \Delta W = W_0 + BA,
\]

where the rank of \(BA\) is much smaller than the dimensions of \(W_0\).

### 3.2 Data Processing

We use the PIPPA-ShareGPT-formatted dataset. Each sample contains a character
profile and a multi-turn human/model conversation. The latest processed data
contains 8,831 training, 1,097 validation, and 758 test conversations.
Individual experiments use at most 4,000 training and 200 validation samples
with seed 42.

The processing pipeline:

- removes invalid, excessively long, and duplicate conversations;
- maps human and model turns to `user` and `assistant`;
- constructs a `system` message from the character name and description;
- splits by conversation/character groups to reduce leakage across splits;
- records file hashes in the training manifest for reproducibility.

### 3.3 Assistant-Only Supervision

The full conversation is rendered with the model's chat template, but loss is
computed only on assistant tokens. Character-card, user, and padding tokens are
assigned label `-100`. This teaches the model to generate role responses
without asking it to reproduce the prompt.

### 3.4 Character-Card-Preserving Truncation

The initial implementation used token-level left truncation. For long samples,
this could remove the character card before removing the target response. The
revised strategy:

1. always retains the complete `system` character card;
2. removes messages after the final supervised assistant response;
3. removes the oldest dialogue messages first;
4. preserves assistant supervision spans and message boundaries;
5. uses the same cropping logic for training and PPL evaluation.

The context limit acts as a soft budget when the character card and final
response alone exceed it.

## 4. Experimental Setup

### 4.1 Training Configuration

All experiments use batch size 1, gradient accumulation 8, FP16,
`paged_adamw_8bit`, cosine scheduling, weight decay 0.01, and 4,000/200
training/validation samples.

| Experiment | Context | Truncation | Learning Rate | Epochs | Main Purpose |
|---|---:|---|---:|---:|---|
| `train_1` | 512 | Token-level left truncation | `2e-4` | 1 | Initial baseline |
| `train_2` | 512 | Preserve character card | `2e-4` | 1 | Truncation ablation |
| `train_3` | 1024 | Preserve character card | `2e-4` | 1 | Context-length ablation |
| `train_4` | 1024 | Preserve character card | `1e-4` | 2 | Lower LR and longer training |

`train_4` evaluates after each epoch and reloads the checkpoint with the lowest
validation loss. Its first-epoch validation loss is 2.359 and its second-epoch
loss is 2.362, so additional training does not improve validation loss.

### 4.2 Evaluation Systems

Each evaluation compares:

- **Base, no card:** base model without character information;
- **Base + card:** base model with the character card as a system prompt;
- **LoRA + card:** LoRA model with the same character card.

Generation is greedy (`do_sample=false`) with at most 256 new tokens. The
evaluation targets 100 single-turn samples and 20 four-turn challenges after
safety, quality, and character-duplication filtering.

### 4.3 Metrics

Automatic metrics include assistant perplexity, Distinct-1/2, repetition rate,
refusal rate, throughput, and GPU memory.

For single-turn evaluation, MiniMax-M3 scores:

| Dimension | Weight |
|---|---:|
| Role identity | 35% |
| Style | 20% |
| Relevance | 20% |
| Naturalness | 15% |
| Immersion and absence of AI meta language | 10% |

The four-turn challenge evaluates role identity, memory, coherence, style, and
immersion. Answers are anonymized, and paired score differences, win rates,
95% confidence intervals, judge success rate, and order consistency are
recorded.

## 5. Results

### 5.1 Training Efficiency

| Experiment | Train Loss | Training Time | Notes |
|---|---:|---:|---|
| `train_1` | 2.364 | 21.3 min | 500 optimizer steps |
| `train_2` | N/A | N/A | Original run manifest unavailable |
| `train_3` | 2.350 | 50.7 min | 1024-token context |
| `train_4` | 2.278 | 101.6 min | 1000 steps, two epochs |

The longest run remains feasible on an RTX 4060 Laptop GPU, but two epochs with
1024-token context require approximately 102 minutes rather than less than one
hour.

### 5.2 LoRA Results Across Experiments

The following table reports the `LoRA + card` system. Judge sample counts are
70 for `train_2` and 99 for `train_3`/`train_4`; cross-run absolute scores
should therefore be interpreted as trends. Within-run paired comparisons
against `Base + card` are more reliable.

| Experiment | PPL ↓ | Identity ↑ | Single Score ↑ | Win Rate vs Base + Card ↑ | Repetition ↓ | Multi Score ↑ |
|---|---:|---:|---:|---:|---:|---:|
| `train_1` | 10.404 | 2.770 | 2.718 | 44.0% | 25.6% | 2.763 |
| `train_2` | **9.441** | 2.657 | 2.643 | 48.6% | 40.2% | 2.225 |
| `train_3` | 10.446 | **2.970** | 2.935 | **55.6%** | 23.9% | **2.890** |
| `train_4` | 10.343 | 2.909 | **2.938** | **55.6%** | **18.1%** | 2.668 |

### 5.3 Single-Turn Role-Play

`train_3` and `train_4` both reach a 55.6% win rate against `Base + card`.
For `train_4`, the weighted score improves from 2.703 to 2.938:

```text
LoRA - Base + card = +0.235
95% CI = [-0.060, 0.558]
```

The interval includes zero. The result indicates an improvement trend, but it
does not provide sufficient evidence for a stable overall advantage. Success
cases show that LoRA can avoid generic safety commentary and respond directly
as the character.

### 5.4 Multi-Turn Role-Play

All four LoRA models underperform the prompted base model in multi-turn
evaluation. For `train_4`:

```text
Base + card weighted score = 3.153
LoRA + card weighted score = 2.668
Paired difference = -0.485
95% CI = [-0.993, -0.005]
```

The LoRA model is competitive in memory but weaker in identity, coherence,
style, and immersion. Longer context improves `train_3` relative to
`train_2`, but increasing training to two epochs does not preserve that gain.

### 5.5 Repetition and Diversity

The lowest PPL belongs to `train_2`, which also has the worst repetition rate
and multi-turn score. This demonstrates that PPL alone is not a valid measure
of role-play quality.

Reducing the learning rate and selecting the best checkpoint in `train_4`
improves Distinct-1/2 to 0.718/0.809 and reduces repetition from 23.9% in
`train_3` to 18.1%. However, the prompted base model has only 2.9% repetition,
so repetitive and template-like generation remains the largest automatic
quality gap.

### 5.6 Effect of the Character Card

The character card itself provides a strong and consistent improvement. In the
`train_4` evaluation, `Base + card` improves over `Base, no card` by 0.765
single-turn weighted points and 1.270 multi-turn points. This gain is larger
and more reliable than the additional gain produced by the current LoRA
models.

### 5.7 Evaluation Reliability

`train_4` records a 99.2% judge success rate and 92.9% order consistency. One
single-turn judge request failed, leaving 99 scored samples. Its base-model
answers were reused from `train_1` after strict configuration and sample
validation, so baseline latency and GPU performance fields are excluded.

## 6. Discussion

The experiments support four conclusions:

1. **LoRA training is effective at distribution adaptation.** All LoRA models
   reduce PPL from approximately 95 for `Base + card` to around 10.
2. **Prompting is already a strong baseline.** Supplying a character card
   greatly improves both single-turn and multi-turn role-play.
3. **Longer context helps more than preserving the card alone.** `train_3`
   recovers much of the quality lost in `train_2`, suggesting that a 512-token
   budget is insufficient for a long card plus useful dialogue context.
4. **Optimization changes reduce repetition but not multi-turn degradation.**
   `train_4` is more diverse than `train_3`, yet its multi-turn score is lower.

The main limitation is likely not a single hyperparameter. Training samples may
contain repetitive expressions, inconsistent narration perspectives, or
truncated user/assistant relationships. Assistant-only SFT can learn these
patterns strongly while still obtaining a low loss.

## 7. Future Optimization Directions

Further work should treat configuration updates as proposed experiments rather
than established improvements.

### 7.1 Data Quality and Truncation

- filter repeated clauses, near-duplicate responses, generic role-play
  templates, and incorrect speaker perspectives;
- guarantee complete `user → assistant` pairs after truncation;
- compare character-balanced sampling with the current dataset sampling;
- create a higher-quality subset and run a controlled `train_5` experiment
  using the `train_4` configuration as the baseline.

### 7.2 Training Configuration

- compare learning rates `5e-5` and `1e-4` with early stopping;
- evaluate one epoch versus the best checkpoint from two epochs;
- test LoRA rank 16 after data cleaning, rather than increasing rank on noisy
  data;
- extend target modules to `gate_proj`, `up_proj`, and `down_proj` in a
  controlled ablation;
- evaluate 1536 or 2048 context only after measuring truncation coverage and
  memory cost;
- consider NEFTune, label smoothing, or a repetition-aware auxiliary objective.

Each experiment should change one principal variable, reuse the same test
samples and baseline generations, and report paired confidence intervals.

### 7.3 Preference Optimization

Construct preference pairs from successful and repetitive outputs, then test
DPO, KTO, or a similar preference-optimization method. Preference data should
reward character fidelity, concise progression, and multi-turn consistency
while penalizing loops and AI meta language.

### 7.4 Evaluation

- add blinded human evaluation by multiple group members;
- repeat judge scoring with swapped answer order or a second judge model;
- enlarge the multi-turn challenge set;
- report confidence intervals and representative failures in the final poster
  and paper.

## 8. Conclusion

QLoRA makes role-play adaptation of Qwen2.5-3B-Instruct feasible on limited
hardware. The resulting models learn the role-play text distribution, reduce
refusals, and show a promising single-turn improvement over the prompted base
model. Nevertheless, the improvement is not statistically conclusive, and
multi-turn role identity, coherence, style, and immersion remain worse than
the strong `Base + card` baseline.

Among the tested configurations, `train_3` gives the best multi-turn LoRA
score, while `train_4` gives the best diversity and lowest repetition. The
next stage should prioritize data cleaning and complete dialogue boundaries,
followed by controlled configuration and preference-optimization experiments.

## References

[1] Hu, E. J., Shen, Y., Wallis, P., et al. "LoRA: Low-Rank Adaptation of
Large Language Models." ICLR 2022.

[2] Dettmers, T., Pagnoni, A., Holtzman, A., and Zettlemoyer, L. "QLoRA:
Efficient Finetuning of Quantized LLMs." NeurIPS 2023.

[3] Qwen Team. "Qwen2.5 Technical Report." 2024.

[4] PIPPA-ShareGPT-formatted dataset:
`KaraKaraWitch/PIPPA-ShareGPT-formatted`.
