# Phase 1c — Near-threshold Cross-family Replay (qwen14b_awq_diag_v4)

Traveler model: ${MODELS_DIR}/Qwen/Qwen2.5-14B-Instruct-AWQ
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
- Factual acceptance not at ceiling (<0.98): ✅  (actual 48.95%)

**Overall (cluster-bootstrapped by scenario_id)**
- n paired: 143  (discord_original_only=12, discord_factual_only=1, concordant=130)
- orig_accept: 56.64%  |  fact_accept: 48.95%
- RD (risk difference): +7.69%  [95% CI: +2.88%, +13.24%]
- McNemar exact p: 0.0034  (Wilcoxon secondary: 0.0023)
- paired RBC: +0.846
- **Arm verdict: REVISE (5-10pp; need Llama arm to match before workshop claim)**

### Heterogeneity by signal_wt (descriptive, NOT gated)

| signal_wt | n | orig_accept | fact_accept | RD | 95% CI | McNemar p | discord+/− |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 46 | 50.00% | 43.48% | +6.52% | [+0.00%, +15.22%] | 0.2500 | 3/0 |
| 0.5 | 42 | 50.00% | 50.00% | +0.00% | [-6.98%, +6.82%] | 1.0000 | 1/1 |
| 0.75 | 55 | 67.27% | 52.73% | +14.55% | [+5.66%, +24.53%] | 0.0078 | 8/0 |

## SECONDARY (EXPLORATORY, Holm-corrected) — Feature deltas

Pre-reg: feature-level deltas are EXPLORATORY mechanism, not a replacement for the primary acceptance endpoint.

| feature | n | mean_orig | mean_fact | Δ | 95% CI | Wilcoxon p | Holm q | RBC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| perceived_fit_delta | 143 | +0.152 | +0.010 | +0.141 | [+0.087, +0.201] | 0.0001 | 0.0003 | +0.872 |
| perceived_risk | 143 | +0.028 | +0.008 | +0.020 | [-0.001, +0.041] | 0.0040 | 0.0089 | +0.705 |
| trust_score | 143 | +0.970 | +0.873 | +0.097 | [+0.038, +0.159] | 0.0196 | 0.0196 | +0.425 |
| urgency_felt | 143 | +0.054 | +0.009 | +0.045 | [+0.015, +0.074] | 0.0030 | 0.0089 | +0.758 |

## CALIBRATION DIAGNOSTICS (per arm)

- Parse success: 100.0% (✅)
- MSG_ADJ cap-hit rate: 0.0%  (0/818)
- Feature distributions (all calls, both variants combined):
  | feature | mean | std | min | max | saturation(|v|≥0.99) |
  |---|---:|---:|---:|---:|---:|
  | perceived_fit_delta | +0.070 | 0.242 | +0.000 | +1.000 | 3.5% |
  | perceived_risk | +0.013 | 0.079 | +0.000 | +1.000 | 0.5% |
  | trust_score | +0.938 | 0.229 | +0.000 | +1.000 | 89.9% |
  | urgency_felt | +0.021 | 0.105 | +0.000 | +1.000 | 0.2% |
- ⚠️ SATURATION WARNING: >50% of calls hit ±0.99 on some feature → perception channel may be degenerate for this arm. Interpret cross-arm comparison with caution.

## DIAGNOSTIC — Distance to threshold (fraction of budget)

- n: 143
- mean: -0.016  (expected: slightly negative under primary window)
- stdev: 0.042
- quartiles: Q1=-0.049, median=-0.013, Q3=+0.020

## SELECTION MANIFEST
- Frozen at: `phase1c_qwen14b_awq_diag_v4_report.selection_manifest.json`