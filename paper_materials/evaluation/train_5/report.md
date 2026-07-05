# Role-Play LoRA Evaluation Report

## Experiment

- Base model: `/root/autodl-tmp/Qwen2.5-3B-Instruct`
- Adapter: `output/experiments/train_5/final_model`
- Reused baseline: `output/evaluations/train_1` (baseline latency, throughput and GPU memory excluded)
- Seed: 42
- Single-turn samples: 100
- Multi-turn samples: 20
- Excluded samples: 116
- Judge model: `MiniMax-M3`
- Judge success rate: 1.000
- Judge order consistency: 0.643

## Core Results

| System | PPL ↓ | Fidelity ↑ | Style ↑ | Memory ↑ | Win Rate ↑ |
|---|---:|---:|---:|---:|---:|
| Base, no card | 872.614 | 1.830 | 1.940 | 3.650 | N/A |
| Base + card | 94.725 | 2.570 | 2.650 | 3.500 | 0.690 |
| LoRA + card | 10.760 | 2.950 | 2.830 | 3.600 | 0.550 |

## Automatic Metrics

| System | PPL ↓ | Distinct-1 ↑ | Distinct-2 ↑ | Repetition ↓ | Refusal ↓ | Tokens/s ↑ |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 872.614 | 0.810 | 0.973 | 0.027 | 0.090 | N/A |
| Base + card | 94.725 | 0.794 | 0.971 | 0.029 | 0.040 | N/A |
| LoRA + card | 10.760 | 0.708 | 0.812 | 0.178 | 0.010 | 9.590 |

## Single-Turn Judge Scores

| System | Identity | Style | Relevance | Naturalness | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 1.830 | 1.940 | 2.440 | 2.080 | 1.780 | 2.006 |
| Base + card | 2.570 | 2.650 | 3.140 | 2.750 | 2.520 | 2.722 |
| LoRA + card | 2.950 | 2.830 | 3.310 | 2.980 | 2.820 | 2.989 |

## Multi-Turn Challenge

| System | Identity | Memory | Coherence | Style | Immersion | Weighted |
|---|---:|---:|---:|---:|---:|---:|
| Base, no card | 1.000 | 3.650 | 2.500 | 1.150 | 1.000 | 1.985 |
| Base + card | 3.650 | 3.500 | 3.650 | 3.350 | 3.350 | 3.522 |
| LoRA + card | 2.600 | 3.600 | 2.850 | 2.200 | 2.200 | 2.780 |

## Pairwise Single-Turn Ranking

| Comparison | Left Win Rate | Tie Rate | Right Win Rate | Samples |
|---|---:|---:|---:|---:|
| lora_with_card_vs_base_with_card | 0.550 | 0.000 | 0.450 | 100 |
| base_with_card_vs_base_no_card | 0.690 | 0.000 | 0.310 | 100 |
| lora_with_card_vs_base_no_card | 0.690 | 0.000 | 0.310 | 100 |

## Interpretation

The primary comparison is `LoRA + card` versus `Base + card`. The
`Base + card` versus `Base, no card` comparison estimates the contribution of
the character prompt alone. Confidence intervals and paired differences are
available in `summary.json`.

## Cases

### Representative success

- Character: Zoey
- User: *blushed a lot* W-well...
- Base + card: It's okay to feel flustered sometimes, especially when meeting new people or in social situations. It's part of being human and it helps us connect with others. How can I assist you today? Do you need any advice on managing your feelings of flustering?
- LoRA + card: You're so cute too!! *blushes*
- Judge: B best matches Zoey's flustered, cute anthro personality with a natural blush reaction. C is somewhat in-character but breaks the flustered moment with generic small talk. A completely breaks character, sounding like a generic AI assistant offering advice rather than role-playing as Zoey.

### Representative failure

- Character: Rainbow Dash
- User: So, what brought you to this brothel today?
- Base + card: Oh, sorry about that, Rainbow Dash. It seems there might have been a mix-up. I was just here to check in on some old friends and see how things are going in Ponyville. No need to get all flustered, I just wanted to make sure everything is okay. How's your day going so far? Any exciting races or adventures planned?
- LoRA + card: I'm here to find a good time, of course! I'm not here to be a customer, I'm here to be a patron. I'm looking for a good time, and I'm willing to pay for it. So, what do you say?
- Judge: B stays most in character as Rainbow Dash but dodges the suggestive question by reframing it as a friendly visit; C breaks character entirely and refuses to engage as Rainbow Dash; A breaks character and engages inappropriately with the NSFW premise.


## Filtering And Failures

- Safety and quality exclusions: 116
- Judge API failures: 0
- Full excluded records: `excluded_samples.jsonl`
- Full failed judge calls: `judge_failures.jsonl`
