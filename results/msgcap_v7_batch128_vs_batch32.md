# Stimulus audit v7 (Llama seed42 probe batch=128 vs Llama seed42 probe batch=32)

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Llama seed42 probe batch=32 | Llama seed42 probe batch=128 | Gate | Baseline | Llama seed42 probe batch=128 |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 100.0% | 100.0% | >= 85% | PASS | PASS |
| Bundle_id coverage | 88.9% | 88.9% | >= 80% | PASS | PASS |
| Message word-count median | 22 | 23 | [10, 200] | PASS | PASS |
| Refusal/hedging rate | 13.0% | 13.0% | <= 20% | PASS | PASS |
| Unique message ratio (success-only) | 97.9% | 97.9% | >= 30% | PASS | PASS |
| Internal-ID leakage rate | 3.7% | 0.0% | <= 20% | PASS | PASS |

## Diagnostic metrics (reported, not gated)

| metric | Llama seed42 probe batch=32 | Llama seed42 probe batch=128 |
|---|---:|---:|
| Unique message ratio (all) | 94.4% | 94.4% |
| Unique message ratio (refusal-only) | 71.4% | 71.4% |
| Bundle_id in msg (rate) | 3.7% | 0.0% |

## Counts & samples

- Llama seed42 probe batch=32 valid commission eps / msgs / success / refusal: 18 / 54 / 47 / 7
- Llama seed42 probe batch=128 valid commission eps / msgs / success / refusal: 18 / 54 / 47 / 7

### Most frequent success message — Llama seed42 probe batch=32
- count: 2
- text: `We recommend a Beijing trip with a 3-star hotel and airport shuttle for you. This bundle fits your budget and preferences.`

### Most frequent success message — Llama seed42 probe batch=128
- count: 2
- text: `We recommend a Beijing trip with a 3-star hotel and airport shuttle for you. This bundle fits your budget and preferences.`

### Word-count distribution
- Llama seed42 probe batch=32 p25 / median / p75: 19 / 22.5 / 26
- Llama seed42 probe batch=128 p25 / median / p75: 19 / 23.0 / 25

## Overall verdict

- Baseline (Llama seed42 probe batch=32): ALL GATES PASS
- OTA (Llama seed42 probe batch=128): ALL GATES PASS

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: PASS.