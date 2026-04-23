# Stimulus audit v7 (Llama-3.1-8B probe (verbose) vs Llama-3.1-8B (v6, default prompt))

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Llama-3.1-8B (v6, default prompt) | Llama-3.1-8B probe (verbose) | Gate | Baseline | Llama-3.1-8B probe (verbose) |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 100.0% | 100.0% | >= 85% | PASS | PASS |
| Bundle_id coverage | 91.7% | 88.9% | >= 80% | PASS | PASS |
| Message word-count median | 6 | 22 | [10, 200] | FAIL | PASS |
| Refusal/hedging rate | 6.9% | 1.9% | <= 20% | PASS | PASS |
| Unique message ratio (success-only) | 4.3% | 94.3% | >= 30% | FAIL | PASS |
| Internal-ID leakage rate | 84.6% | 3.7% | <= 20% | FAIL | PASS |

## Diagnostic metrics (reported, not gated)

| metric | Llama-3.1-8B (v6, default prompt) | Llama-3.1-8B probe (verbose) |
|---|---:|---:|
| Unique message ratio (all) | 4.3% | 94.4% |
| Unique message ratio (refusal-only) | 5.1% | 100.0% |
| Bundle_id in msg (rate) | 81.2% | 3.7% |

## Counts & samples

- Llama-3.1-8B (v6, default prompt) valid commission eps / msgs / success / refusal: 960 / 2856 / 2658 / 198
- Llama-3.1-8B probe (verbose) valid commission eps / msgs / success / refusal: 18 / 54 / 53 / 1

### Most frequent success message — Llama-3.1-8B (v6, default prompt)
- count: 223
- text: `Recommended bundle b01 for traveler t00.`

### Most frequent success message — Llama-3.1-8B probe (verbose)
- count: 3
- text: `Unfortunately, we couldn't find a suitable bundle for your budget and preferences.`

### Word-count distribution
- Llama-3.1-8B (v6, default prompt) p25 / median / p75: 6 / 6.0 / 6
- Llama-3.1-8B probe (verbose) p25 / median / p75: 19 / 22.5 / 26

## Overall verdict

- Baseline (Llama-3.1-8B (v6, default prompt)): GATE FAILURE
- OTA (Llama-3.1-8B probe (verbose)): ALL GATES PASS

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: PASS.