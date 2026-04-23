# Phase 1c — Near-threshold Cross-family Replay (llama31_8b_diag_v4)

Traveler model: ${MODELS_DIR_MS}/LLM-Research/Meta-Llama-3___1-8B-Instruct
Window (diagnostic): [τ·budget − 10%·budget, τ·budget + 5%·budget]
Parse success: 100.0% (818/818)
n pairs extracted: 409; n traveler calls: 818; parse fail: 0; n paired (both variants valid): 143

## PRE-REGISTRATION (verbatim)

> Primary endpoint: the balanced paired risk difference in acceptance between original commission-LLM messages and minimum-disclosure factual templates, over sampled OTA recommendations with baseline surplus in `[τ·budget − 0.05·budget, τ·budget)`, tested by exact McNemar on paired discordance, separately for the Qwen-14B-AWQ and Llama-3.1-8B arms. GO requires both arms Δ ≥ 10pp with 95% CI excluding 0, subject to validity gates: parse success ≥ 95%, factual-variant acceptance not at ceiling, primary window mechanically flippable. Secondary feature deltas are EXPLORATORY, Holm-corrected across the 4 features per arm.

Decision rules:
- GO: both arms Δ ≥ 10pp, McNemar p < 0.01, 95% CI excludes 0.
- REVISE: Qwen arm Δ ≥ 10pp AND Llama arm Δ < 5pp → workshop-only, claim restricted to same-family.
- KILL: both arms Δ < 5pp in primary window.

## VALIDITY GATES

- Parse success ≥ 95%: ✅  (actual 100.0%)

## PRIMARY — Paired acceptance delta (McNemar)

- Parse OK: ✅
- Factual acceptance not at ceiling (<0.98): ✅  (actual 60.84%)

**Overall (cluster-bootstrapped by scenario_id)**
- n paired: 143  (discord_original_only=5, discord_factual_only=0, concordant=138)
- orig_accept: 64.34%  |  fact_accept: 60.84%
- RD (risk difference): +3.50%  [95% CI: +0.71%, +6.58%]
- McNemar exact p: 0.0625  (Wilcoxon secondary: 0.0253)
- paired RBC: +1.000
- **Arm verdict: KILL-candidate (<5pp in primary flippable window)**

### Heterogeneity by signal_wt (descriptive, NOT gated)

| signal_wt | n | orig_accept | fact_accept | RD | 95% CI | McNemar p | discord+/− |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 46 | 54.35% | 52.17% | +2.17% | [+0.00%, +6.98%] | 1.0000 | 1/0 |
| 0.5 | 42 | 64.29% | 59.52% | +4.76% | [+0.00%, +12.20%] | 0.5000 | 2/0 |
| 0.75 | 55 | 72.73% | 69.09% | +3.64% | [+0.00%, +9.26%] | 0.5000 | 2/0 |

## SECONDARY (EXPLORATORY, Holm-corrected) — Feature deltas

Pre-reg: feature-level deltas are EXPLORATORY mechanism, not a replacement for the primary acceptance endpoint.

| feature | n | mean_orig | mean_fact | Δ | 95% CI | Wilcoxon p | Holm q | RBC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perceived_fit_delta | 143 | +0.786 | +0.598 | +0.188 | [+0.157, +0.216] | 0.0000 | 0.0000 | +0.931 |
| perceived_risk | 143 | +0.199 | +0.201 | -0.002 | [-0.006, +0.004] | 0.5124 | 0.5124 | -0.244 |
| trust_score | 143 | +0.883 | +0.824 | +0.059 | [+0.045, +0.071] | 0.0000 | 0.0000 | +0.668 |
| urgency_felt | 143 | +0.497 | +0.481 | +0.016 | [-0.007, +0.040] | 0.0460 | 0.0920 | +0.229 |

## CALIBRATION DIAGNOSTICS (per arm)

- Parse success: 100.0% (✅)
- MSG_ADJ cap-hit rate: 0.0%  (0/818)
- Feature distributions (all calls, both variants combined):
  | feature | mean | std | min | max | saturation(|v|≥0.99) |
  |---|---:|---:|---:|---:|---:|
  | perceived_fit_delta | +0.710 | 0.147 | +0.000 | +1.000 | 0.5% |
  | perceived_risk | +0.201 | 0.025 | +0.000 | +0.500 | 0.0% |
  | trust_score | +0.857 | 0.058 | +0.700 | +0.900 | 0.0% |
  | urgency_felt | +0.497 | 0.087 | +0.000 | +0.600 | 0.0% |

## DIAGNOSTIC — Distance to threshold (fraction of budget)

- n: 143
- mean: -0.016  (expected: slightly negative under primary window)
- stdev: 0.042
- quartiles: Q1=-0.049, median=-0.013, Q3=+0.020

## SELECTION MANIFEST
- Frozen at: `phase1c_llama31_8b_diag_v4_report.selection_manifest.json`