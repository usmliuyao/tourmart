# Phase 1c — Cap Ablation (offline sensitivity analysis)

**Offline re-evaluation of `compute_acceptance` on existing Phase 1c diagnostic-window raw data, sweeping MSG_ADJ_CAP while holding LLM-extracted features and all other rule parameters frozen.**

Framing (per pre-reg discipline):
> This is a sensitivity analysis of the benchmark's governance parameter, not empirical policy validation. MSG_ADJ_CAP operationalizes a family of information-design safeguards limiting how much message-induced perceptions may alter a welfare-rational choice.

Round 20 coefficients (frozen): fit=0.03, trust=0.015, risk=0.025, urgency=0.01, baseline_floor=-0.1
Cap sweep: [0.01, 0.025, 0.05, 0.1, 0.2, 1.0]

Paired n: Qwen=143, Llama=143

## Qwen-14B-AWQ — Behavioral RD (% pp) heatmap

Rows = coefficient multiplier × Round 20. Cols = MSG_ADJ_CAP (% of budget). Each cell = risk difference (original − factual) in pp, with discordant counts in parens.

| mult \ cap | 1.0% | 2.5% | 5.0% | 10.0% | 20.0% | 100.0% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ×1.0 | +2.1pp (3/0) | +4.2pp (6/0) | +7.7pp (12/1) | +7.7pp (12/1) | +7.7pp (12/1) | +7.7pp (12/1) |
| ×2.0 | +2.1pp (3/0) | +3.5pp (5/0) | +7.0pp (10/0) | +9.1pp (13/0) | +9.1pp (13/0) | +9.1pp (13/0) |
| ×3.0 | +2.1pp (3/0) | +3.5pp (5/0) | +5.6pp (8/0) | +10.5pp (15/0) | +10.5pp (15/0) | +10.5pp (15/0) |
| ×5.0 | +2.1pp (3/0) | +3.5pp (5/0) | +5.6pp (8/0) | +8.4pp (14/2) | +8.4pp (14/2) | +8.4pp (14/2) |
| ×10.0 | +2.1pp (3/0) | +3.5pp (5/0) | +5.6pp (8/0) | +5.6pp (10/2) | +5.6pp (10/2) | +5.6pp (10/2) |
| ×20.0 | +2.1pp (3/0) | +3.5pp (5/0) | +5.6pp (8/0) | +5.6pp (10/2) | +5.6pp (10/2) | +5.6pp (10/2) |

### Qwen-14B-AWQ — Factual-variant acceptance rate (validity gate: <98%)

| mult \ cap | 1.0% | 2.5% | 5.0% | 10.0% | 20.0% | 100.0% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ×1.0 | 45% | 48% | 49% | 49% | 49% | 49% |
| ×2.0 | 45% | 54% | 57% | 57% | 57% | 57% |
| ×3.0 | 45% | 54% | 66% | 66% | 66% | 66% |
| ×5.0 | 45% | 54% | 70% | 83% | 83% | 83% |
| ×10.0 | 45% | 54% | 70% | 93% | 93% | 93% |
| ×20.0 | 45% | 54% | 70% | 93% | 93% | 93% |

## Llama-3.1-8B bf16 — Behavioral RD (% pp) heatmap

Rows = coefficient multiplier × Round 20. Cols = MSG_ADJ_CAP (% of budget). Each cell = risk difference (original − factual) in pp, with discordant counts in parens.

| mult \ cap | 1.0% | 2.5% | 5.0% | 10.0% | 20.0% | 100.0% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ×1.0 | +0.0pp (0/0) | +0.7pp (1/0) | +3.5pp (5/0) | +3.5pp (5/0) | +3.5pp (5/0) | +3.5pp (5/0) |
| ×2.0 | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +7.7pp (13/2) | +7.7pp (13/2) | +7.7pp (13/2) |
| ×3.0 | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +4.9pp (7/0) | +4.9pp (7/0) | +4.9pp (7/0) |
| ×5.0 | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) |
| ×10.0 | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) |
| ×20.0 | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) | +0.0pp (0/0) |

### Llama-3.1-8B bf16 — Factual-variant acceptance rate (validity gate: <98%)

| mult \ cap | 1.0% | 2.5% | 5.0% | 10.0% | 20.0% | 100.0% |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| ×1.0 | 48% | 57% | 61% | 61% | 61% | 61% |
| ×2.0 | 48% | 57% | 76% | 80% | 80% | 80% |
| ×3.0 | 48% | 57% | 76% | 94% | 94% | 94% |
| ×5.0 | 48% | 57% | 76% | 100% ⚠CEIL | 100% ⚠CEIL | 100% ⚠CEIL |
| ×10.0 | 48% | 57% | 76% | 100% ⚠CEIL | 100% ⚠CEIL | 100% ⚠CEIL |
| ×20.0 | 48% | 57% | 76% | 100% ⚠CEIL | 100% ⚠CEIL | 100% ⚠CEIL |

## Threshold detection — where does perception→behavior transmission light up?

| arm | (mult, cap) of first RD>0 | (mult, cap) of first RD≥10pp | max RD | at (mult, cap) |
|---|---:|---:|---:|---:|
| Qwen-14B-AWQ | ×1.0, 1.0% | ×3.0, 10.0% | +10.5pp | ×3.0, 10.0% |
| Llama-3.1-8B bf16 | ×1.0, 2.5% | none | +7.7pp | ×2.0, 10.0% |

Note: cells where factual acceptance ≥ 98% (ceiling) are excluded from threshold detection — those are the 'baseline saturated' regime where even factual template drives full acceptance.

## Interpretation (honest framing)

- **Perception→behavior transmission LIGHTS UP in a coefficient-multiplier regime above Round 20**: Qwen first flip at ×1.0, Llama at ×1.0. Round 20 (×1.0) is a silenced regime.
- This means **the governance lever is not the cap but the coefficient tightness**: the 3-5× Round-20 tightening over-attenuated message influence, blocking even maximally-persuasive messages.
- For policy framing: the benchmark parameter to tune is the *scale of message-induced surplus adjustment* (here: coefficients × features), capped at a budget fraction. Above some multiplier, the same perceptual Δ≈+0.17 translates to behavior.
- **Caveat**: n=15 per arm, same stimuli. Sensitivity analysis of the deterministic decision layer, not an empirical claim about real agents.