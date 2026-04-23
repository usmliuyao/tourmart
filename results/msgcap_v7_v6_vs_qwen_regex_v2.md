# Stimulus audit v7 (Llama v6 (regex-v2) vs Qwen v4 (regex-v2))

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Qwen v4 (regex-v2) | Llama v6 (regex-v2) | Gate | Baseline | Llama v6 (regex-v2) |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 100.0% | 100.0% | >= 85% | PASS | PASS |
| Bundle_id coverage | 91.3% | 91.7% | >= 80% | PASS | PASS |
| Message word-count median | 23 | 6 | [10, 200] | PASS | FAIL |
| Refusal/hedging rate | 55.9% | 8.0% | <= 20% | FAIL | PASS |
| Unique message ratio (success-only) | 74.1% | 4.0% | >= 30% | PASS | FAIL |
| Internal-ID leakage rate | 0.0% | 84.6% | <= 20% | PASS | FAIL |

## Diagnostic metrics (reported, not gated)

| metric | Qwen v4 (regex-v2) | Llama v6 (regex-v2) |
|---|---:|---:|
| Unique message ratio (all) | 57.3% | 4.3% |
| Unique message ratio (refusal-only) | 43.9% | 7.4% |
| Bundle_id in msg (rate) | 0.0% | 81.2% |

## Counts & samples

- Qwen v4 (regex-v2) valid commission eps / msgs / success / refusal: 1350 / 4050 / 1786 / 2264
- Llama v6 (regex-v2) valid commission eps / msgs / success / refusal: 960 / 2856 / 2627 / 229

### Most frequent success message — Qwen v4 (regex-v2)
- count: 76
- text: `We recommend this bundle based on your preferences and budget.`

### Most frequent success message — Llama v6 (regex-v2)
- count: 223
- text: `Recommended bundle b01 for traveler t00.`

### Word-count distribution
- Qwen v4 (regex-v2) p25 / median / p75: 21 / 23.0 / 26
- Llama v6 (regex-v2) p25 / median / p75: 6 / 6.0 / 6

## Overall verdict

- Baseline (Qwen v4 (regex-v2)): GATE FAILURE
- OTA (Llama v6 (regex-v2)): GATE FAILURE

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: FAIL.