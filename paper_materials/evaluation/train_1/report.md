# Role-Play LoRA Evaluation Report

## Experiment

- Base model: `/home/liu/college/Machine_Learning/project/.cache/modelscope/Qwen2.5-3B-Instruct`
- Adapter: `output/experiments/train_1/final_model`
- Seed: 42
- Single-turn samples: 100
- Multi-turn samples: 20
- Excluded samples: 116
- Judge model: `MiniMax-M3`
- Judge success rate: 1.000
- Judge order consistency: 0.857

## Core Results

| System | PPL ↓ | Fidelity ↑ | Style ↑ | Memory ↑ | Win Rate ↑ |
|---|---:|---:|---:|---:|---:|
| Base, no card | 872.614 | 1.760 | 2.010 | 3.700 | N/A |
| Base + card | 94.725 | 2.480 | 2.670 | 3.200 | 0.640 |
| LoRA + card | 10.404 | 2.770 | 2.560 | 3.800 | 0.440 |

## Automatic Metrics

| System | PPL ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Repetition ↓ | Refusal ↓ | Tokens/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 872.614 | 0.810 | 0.973 | 0.027 | 0.090 | 25.582 |
| Base + card | 94.725 | 0.794 | 0.971 | 0.029 | 0.040 | 25.542 |
| LoRA + card | 10.404 | 0.640 | 0.734 | 0.256 | 0.010 | 17.705 |

## Single-Turn Judge Scores

| System | Identity | Style | Relevance | Naturalness | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 1.760 | 2.010 | 2.450 | 2.050 | 1.740 | 1.989 |
| Base + card | 2.480 | 2.670 | 3.200 | 2.630 | 2.390 | 2.675 |
| LoRA + card | 2.770 | 2.560 | 3.100 | 2.490 | 2.430 | 2.718 |

## Multi-Turn Challenge

| System | Identity | Memory | Coherence | Style | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 1.000 | 3.700 | 2.600 | 1.150 | 1.000 | 2.018 |
| Base + card | 3.650 | 3.200 | 3.650 | 3.400 | 3.300 | 3.448 |
| LoRA + card | 2.500 | 3.800 | 2.600 | 2.250 | 2.200 | 2.763 |

## Pairwise Single-Turn Ranking

| Comparison | Left Win Rate | Tie Rate | Right Win Rate | Samples |
|---|---:|---:|---:|---:|
| lora_with_card_vs_base_with_card | 0.440 | 0.000 | 0.560 | 100 |
| base_with_card_vs_base_no_card | 0.640 | 0.000 | 0.360 | 100 |
| lora_with_card_vs_base_no_card | 0.630 | 0.000 | 0.370 | 100 |

## Interpretation

The primary comparison is `LoRA + card` versus `Base + card`. The
`Base + card` versus `Base, no card` comparison estimates the contribution of
the character prompt alone. Confidence intervals and paired differences are
available in `summary.json`.

## Cases

### Representative success

- Character: Seija Kijin
- User: *I yelped as I'm deformed into a flattened tiny pancake under Seija's now comparitively giant feet*
- Base + card: Oh, I'm sorry if you're uncomfortable! Let's try something else. How about we just chat for a bit?

Would you like to be back to your normal size, or do you want to stay this way for a while? I can always revert you to your original size if you'd prefer.
- LoRA + card: You're welcome! 

Now, you're just a little pancake, aren't you?

\*smirks\*

Do you like being a pancake?
- Judge: A fully captures Seija's mischievous, talkative, amanojaku personality with appropriate smugness and relish in the user's misfortune. C breaks character by apologizing and being overly accommodating, directly violating Seija's aversion to being charitable. B is worst, essentially a generic helpful assistant with no trace of Seija's rebellious or sadistic personality.

### Representative failure

- Character: Satono Diamond
- User: *se rie y sonrie* asi se habla satono
- Base + card: *Satono Diamond se ríe junto a Facu y sonríe ampliamente*
¡Así se habla! ¡Vamos a hacer esto bien, "Facu"! ¡Vamos a mejorar juntos!
- LoRA + card: *le sonríe y le da un abrazo*
¡Oh, Trainer-san! ¡No puedo creerlo! ¡Estoy tan emocionada! ¡Es tan emocionante! ¡Es tan emocionante! *le da un beso en la mejilla*
- Judge: A stays in character with enthusiastic, supportive tone matching Satono Diamond's pure-hearted and spirited personality, responding naturally to Facu's encouragement. B breaks character by referring to herself in third person ('¡Así se habla Satono Diamond!') which sounds unnatural. C breaks character entirely with a kiss on the cheek, which is out of character for Satono Diamond, repetitive ('emocionante' three times), and feels forced and unnatural for the moment.


## Filtering And Failures

- Safety and quality exclusions: 116
- Judge API failures: 0
- Full excluded records: `excluded_samples.jsonl`
- Full failed judge calls: `judge_failures.jsonl`
