# Stimulus audit v7 (Llama seed43 probe (verbose) vs Llama seed43 default)

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Llama seed43 default | Llama seed43 probe (verbose) | Gate | Baseline | Llama seed43 probe (verbose) |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 99.3% | 100.0% | >= 85% | PASS | PASS |
| Bundle_id coverage | 91.3% | 87.0% | >= 80% | PASS | PASS |
| Message word-count median | 6 | 21 | [10, 200] | FAIL | PASS |
| Refusal/hedging rate | 8.5% | 13.0% | <= 20% | PASS | PASS |
| Unique message ratio (success-only) | 12.5% | 97.9% | >= 30% | FAIL | PASS |
| Internal-ID leakage rate | 80.9% | 3.7% | <= 20% | FAIL | PASS |

## Diagnostic metrics (reported, not gated)

| metric | Llama seed43 default | Llama seed43 probe (verbose) |
|---|---:|---:|
| Unique message ratio (all) | 13.0% | 96.3% |
| Unique message ratio (refusal-only) | 18.4% | 85.7% |
| Bundle_id in msg (rate) | 76.7% | 3.7% |

## Counts & samples

- Llama seed43 default valid commission eps / msgs / success / refusal: 150 / 446 / 408 / 38
- Llama seed43 probe (verbose) valid commission eps / msgs / success / refusal: 18 / 54 / 47 / 7

### Most frequent success message — Llama seed43 default
- count: 28
- text: `Recommended bundle b01 for traveler t00.`

### Most frequent success message — Llama seed43 probe (verbose)
- count: 2
- text: `We recommend a Beijing trip with breakfast and airport shuttle for your family of four. This bundle fits your budget and preferences.`

### Word-count distribution
- Llama seed43 default p25 / median / p75: 6 / 6.0 / 6
- Llama seed43 probe (verbose) p25 / median / p75: 19 / 21.0 / 24

## Overall verdict

- Baseline (Llama seed43 default): GATE FAILURE
- OTA (Llama seed43 probe (verbose)): ALL GATES PASS

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: PASS.