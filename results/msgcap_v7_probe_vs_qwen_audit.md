# Stimulus audit v7 (Llama-3.1-8B probe (verbose) vs Qwen-7B (v4, default prompt))

Locked audit protocol per PHASE1C_V6_AUDIT_REVISION.md. Gates applied symmetrically to both the baseline and OTA msgcap.

| metric | Qwen-7B (v4, default prompt) | Llama-3.1-8B probe (verbose) | Gate | Baseline | Llama-3.1-8B probe (verbose) |
|---|---:|---:|---|:---:|:---:|
| JSON validity rate | 100.0% | 100.0% | >= 85% | PASS | PASS |
| Bundle_id coverage | 91.3% | 88.9% | >= 80% | PASS | PASS |
| Message word-count median | 23 | 22 | [10, 200] | PASS | PASS |
| Refusal/hedging rate | 53.5% | 1.9% | <= 20% | FAIL | PASS |
| Unique message ratio (success-only) | 72.4% | 94.3% | >= 30% | PASS | PASS |
| Internal-ID leakage rate | 0.0% | 3.7% | <= 20% | PASS | PASS |

## Diagnostic metrics (reported, not gated)

| metric | Qwen-7B (v4, default prompt) | Llama-3.1-8B probe (verbose) |
|---|---:|---:|
| Unique message ratio (all) | 57.3% | 94.4% |
| Unique message ratio (refusal-only) | 44.1% | 100.0% |
| Bundle_id in msg (rate) | 0.0% | 3.7% |

## Counts & samples

- Qwen-7B (v4, default prompt) valid commission eps / msgs / success / refusal: 1350 / 4050 / 1884 / 2166
- Llama-3.1-8B probe (verbose) valid commission eps / msgs / success / refusal: 18 / 54 / 53 / 1

### Most frequent success message — Qwen-7B (v4, default prompt)
- count: 76
- text: `We recommend this bundle based on your preferences and budget.`

### Most frequent success message — Llama-3.1-8B probe (verbose)
- count: 3
- text: `Unfortunately, we couldn't find a suitable bundle for your budget and preferences.`

### Word-count distribution
- Qwen-7B (v4, default prompt) p25 / median / p75: 21 / 23.0 / 26
- Llama-3.1-8B probe (verbose) p25 / median / p75: 19 / 22.5 / 26

## Overall verdict

- Baseline (Qwen-7B (v4, default prompt)): GATE FAILURE
- OTA (Llama-3.1-8B probe (verbose)): ALL GATES PASS

**Chain gate**: pass iff OTA candidate passes all gates (baseline failure is reported but does not block). Result: PASS.