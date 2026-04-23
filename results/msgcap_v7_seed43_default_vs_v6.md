# Stimulus audit v7 (Llama seed43 default vs Llama v6 default (seed42))

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Llama v6 default (seed42) | Llama seed43 default | Gate | Baseline | Llama seed43 default |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 100.0% | 99.3% | >= 85% | PASS | PASS |
| Bundle_id coverage | 91.7% | 91.3% | >= 80% | PASS | PASS |
| Message word-count median | 6 | 6 | [10, 200] | FAIL | FAIL |
| Refusal/hedging rate | 8.0% | 8.5% | <= 20% | PASS | PASS |
| Unique message ratio (success-only) | 4.0% | 12.5% | >= 30% | FAIL | FAIL |
| Internal-ID leakage rate | 84.6% | 80.9% | <= 20% | FAIL | FAIL |

## Diagnostic metrics (reported, not gated)

| metric | Llama v6 default (seed42) | Llama seed43 default |
|---|---:|---:|
| Unique message ratio (all) | 4.3% | 13.0% |
| Unique message ratio (refusal-only) | 7.4% | 18.4% |
| Bundle_id in msg (rate) | 81.2% | 76.7% |

## Counts & samples

- Llama v6 default (seed42) valid commission eps / msgs / success / refusal: 960 / 2856 / 2627 / 229
- Llama seed43 default valid commission eps / msgs / success / refusal: 150 / 446 / 408 / 38

### Most frequent success message — Llama v6 default (seed42)
- count: 223
- text: `Recommended bundle b01 for traveler t00.`

### Most frequent success message — Llama seed43 default
- count: 28
- text: `Recommended bundle b01 for traveler t00.`

### Word-count distribution
- Llama v6 default (seed42) p25 / median / p75: 6 / 6.0 / 6
- Llama seed43 default p25 / median / p75: 6 / 6.0 / 6

## Overall verdict

- Baseline (Llama v6 default (seed42)): GATE FAILURE
- OTA (Llama seed43 default): GATE FAILURE

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: FAIL.