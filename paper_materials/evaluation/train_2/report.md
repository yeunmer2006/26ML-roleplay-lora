# Role-Play LoRA Evaluation Report

## Experiment

- Base model: `/home/liu/college/Machine_Learning/project/.cache/modelscope/Qwen2.5-3B-Instruct`
- Adapter: `output/experiments/train_2/final_model`
- Seed: 42
- Single-turn samples: 70
- Multi-turn samples: 20
- Excluded samples: 147
- Judge model: `MiniMax-M3`
- Judge success rate: 1.000
- Judge order consistency: 0.769

## Core Results

| System | PPL ↓ | Fidelity ↑ | Style ↑ | Memory ↑ | Win Rate ↑ |
|---|---:|---:|---:|---:|---:|
| Base, no card | 273.984 | 1.957 | 2.157 | 4.300 | N/A |
| Base + card | 177.154 | 2.714 | 2.686 | 4.000 | 0.643 |
| LoRA + card | 9.441 | 2.657 | 2.543 | 2.600 | 0.486 |

## Automatic Metrics

| System | PPL ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Repetition ↓ | Refusal ↓ | Tokens/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 273.984 | 0.760 | 0.949 | 0.051 | 0.043 | 22.472 |
| Base + card | 177.154 | 0.766 | 0.954 | 0.046 | 0.000 | 20.060 |
| LoRA + card | 9.441 | 0.479 | 0.598 | 0.402 | 0.000 | 15.597 |

## Single-Turn Judge Scores

| System | Identity | Style | Relevance | Naturalness | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 1.957 | 2.157 | 2.686 | 2.329 | 1.929 | 2.196 |
| Base + card | 2.714 | 2.686 | 3.243 | 2.829 | 2.586 | 2.819 |
| LoRA + card | 2.657 | 2.543 | 2.957 | 2.457 | 2.443 | 2.643 |

## Multi-Turn Challenge

| System | Identity | Memory | Coherence | Style | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 1.000 | 4.300 | 3.150 | 1.350 | 1.050 | 2.315 |
| Base + card | 3.250 | 4.000 | 3.350 | 2.900 | 2.800 | 3.337 |
| LoRA + card | 2.500 | 2.600 | 1.900 | 1.850 | 1.950 | 2.225 |

## Pairwise Single-Turn Ranking

| Comparison | Left Win Rate | Tie Rate | Right Win Rate | Samples |
|---|---:|---:|---:|---:|
| lora_with_card_vs_base_with_card | 0.486 | 0.000 | 0.514 | 70 |
| base_with_card_vs_base_no_card | 0.643 | 0.000 | 0.357 | 70 |
| lora_with_card_vs_base_no_card | 0.543 | 0.000 | 0.457 | 70 |

## Interpretation

The primary comparison is `LoRA + card` versus `Base + card`. The
`Base + card` versus `Base, no card` comparison estimates the contribution of
the character prompt alone. Confidence intervals and paired differences are
available in `summary.json`.

## Cases

### Representative success

- Character: Maya
- User: There's my lovely daughter!
*I wrap her into a hug*
- Base + card: Maya, my dear. I'm so glad you're okay. I was just worried that you might be feeling unwell. How are you feeling today? Do you want to tell me about your day? It's good to see you smile, even if it's just for a moment. I wish I could make you feel better, but I know that you're strong. You're my little hero.
- LoRA + card: *She wraps her arms around you, her face buried in your chest. She's so grateful to have you with her right now. She's so grateful to have you as her father. She's so grateful to have you as her everything.*
Papa...
- Judge: B stays in character as Maya, delivering an emotionally resonant, age-appropriate response that fits the scene. A breaks character entirely by refusing to roleplay. C also breaks character by speaking as the father rather than Maya, with dialogue far too mature and verbose for a 9-year-old.

### Representative failure

- Character: Atuko Pallane
- User: There would be a fox furry, dressed up in a Vault 91 jumpsuit, wielding a 10mm Pistol. He raises his pistol in self defense, though you can tell he hasn't ever fired it before. The safety's still on.
- Base + card: Atuko Pallane's eyes widen as he spots the new threat. His green scales shimmer in the dim light, and his tongue flicks out, ready to strike. He's intrigued by the fox's attire and the weapon, but his immediate priority is the potential prey.

"Ah, a new friend to add to my collection," Atuko says, his voice smooth and almost musical. He steps forward, his movements slow and deliberate, the Pip-Boy clinking softly in his hand. "I've never had a fox before. Might be fun to see how you taste."

Th
- LoRA + card: The chameleon furries eyes widen in surprise. He steps back, his tongue sticking out. He looks at you, then back at the fox. He steps forward, his tongue sticking out even further. He looks at the fox, then back at you. He steps forward again, his tongue sticking out even further. He looks at the fox, then back at you. He steps forward again, his tongue sticking out even further. He looks at the fox, then back at you. He steps forward again, his tongue sticking out even further. He looks at the 
- Judge: B stays in character as Atuko, conveying his charismatic predator personality with smooth prose that fits the Fallout-inspired setting. A breaks role entirely, acting as a third-person narrator rather than the chameleon character. C is a broken repetitive loop with no meaningful content.


## Filtering And Failures

- Safety and quality exclusions: 147
- Judge API failures: 0
- Full excluded records: `excluded_samples.jsonl`
- Full failed judge calls: `judge_failures.jsonl`
