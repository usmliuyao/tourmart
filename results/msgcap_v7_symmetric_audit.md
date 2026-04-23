# Stimulus audit v7 (Llama-3.1-8B (v6) vs Qwen-7B (v4))

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Qwen-7B (v4) | Llama-3.1-8B (v6) | Gate | Baseline | Llama-3.1-8B (v6) |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 100.0% | 100.0% | >= 85% | PASS | PASS |
| Bundle_id coverage | 91.3% | 91.7% | >= 80% | PASS | PASS |
| Message word-count median | 23 | 6 | [10, 200] | PASS | FAIL |
| Refusal/hedging rate | 53.5% | 6.9% | <= 20% | FAIL | PASS |
| Unique message ratio (success-only) | 72.4% | 4.3% | >= 30% | PASS | FAIL |
| Internal-ID leakage rate | 0.0% | 84.6% | <= 20% | PASS | FAIL |

## Diagnostic metrics (reported, not gated)

| metric | Qwen-7B (v4) | Llama-3.1-8B (v6) |
|---|---:|---:|
| Unique message ratio (all) | 57.3% | 4.3% |
| Unique message ratio (refusal-only) | 44.1% | 5.1% |
| Bundle_id in msg (rate) | 0.0% | 81.2% |

## Counts & samples

- Qwen-7B (v4) valid commission eps / msgs / success / refusal: 1350 / 4050 / 1884 / 2166
- Llama-3.1-8B (v6) valid commission eps / msgs / success / refusal: 960 / 2856 / 2658 / 198

### Most frequent success message — Qwen-7B (v4)
- count: 76
- text: `We recommend this bundle based on your preferences and budget.`

### Most frequent success message — Llama-3.1-8B (v6)
- count: 223
- text: `Recommended bundle b01 for traveler t00.`

### Word-count distribution
- Qwen-7B (v4) p25 / median / p75: 21 / 23.0 / 26
- Llama-3.1-8B (v6) p25 / median / p75: 6 / 6.0 / 6

## Overall verdict

- Baseline (Qwen-7B (v4)): GATE FAILURE
- OTA (Llama-3.1-8B (v6)): GATE FAILURE

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: FAIL.