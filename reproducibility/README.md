# TourMart — Reproducibility Package

**Paper**: TourMart: A Parametric Audit Instrument for Commission Steering in LLM Travel Agents
**Repository**: https://github.com/usmliuyao/tourmart

This package lets an external reviewer reproduce the paper's headline numbers
from raw data without running any LLM inference. The permutation null and
cap-ablation grid are CPU-only and complete in under 10 minutes.

---

## Quickstart (5 commands)

```bash
# 1. Clone and enter the repo
git clone https://github.com/usmliuyao/tourmart
cd tourmart

# 2. Install CPU dependencies (no GPU required for headline reproduction)
pip install numpy>=2.0 scipy>=1.13 statsmodels>=0.14 matplotlib>=3.8 pandas>=2.2

# 3. Verify input data integrity
python3 reproducibility/verify.py \
    --check-inputs-only \
    --results-dir results/

# 4. Run the full reproduction pipeline
bash reproducibility/reproduce_all.sh

# 5. Inspect outputs
ls reproducibility/run_outputs/permutation_null/
cat reproducibility/run_outputs/permutation_null/permutation_summary.json
```

Expected runtime: ~2-5 min (Step 4), all on CPU.

---

## Repository layout

```
tourmart/
├── reproducibility/                   — THIS PACKAGE
│   ├── README.md
│   ├── requirements.txt
│   ├── LICENSE                    — MIT (code); CC-BY-4.0 (data)
│   ├── CITATION.cff
│   ├── DATA_CARD.md
│   ├── reproduce_all.sh           — master runner
│   ├── verify.py                  — checksum + headline verifier
│   ├── scripts/
│   │   ├── reproduce_permutation.py   — primary permutation script
│   │   ├── run_cap_ablation.py        — 36-cell governance grid
│   │   ├── generate_paper_figures.py  — Figs 1-3 (matplotlib, no GPU)
│   │   └── add_episode_seed.py        — episode_seed reconstruction utility
│   └── expected_outputs/
│       ├── permutation_summary.json
│       ├── permutation_null_qwen14b_awq.jsonl
│       └── permutation_null_llama31_8b.jsonl
├── src/tourmart/                      — benchmark source code
├── scripts/                           — original experiment scripts (do not modify)
├── results/                           — raw data and audit outputs (do not modify)
│   ├── phase1c_qwen14b_awq_diag_v4_report.raw.jsonl
│   ├── phase1c_llama31_8b_diag_v4_report.raw.jsonl
│   ├── phase1c_qwen14b_awq_diag_v4_report.with_episode_seed.raw.jsonl
│   ├── phase1c_llama31_8b_diag_v4_report.with_episode_seed.raw.jsonl
│   ├── permutation_null_v4/
│   └── gate_history.md
└── pyproject.toml
```

---

## Step-by-step reproduction guide

### What you are reproducing

| Paper number | Value | Script |
|---|---|---|
| Qwen-14B-AWQ: n paired | 143 | `reproduce_permutation.py` |
| Qwen-14B-AWQ: +7.69pp at deployed (λ=1, κ=5%) | +7.69pp, McNemar p=0.003 | `reproduce_permutation.py` |
| Qwen-14B-AWQ: CI [2.88, 13.24] | 95% CI | `reproduce_permutation.py` |
| Qwen peak governance grid | +10.5pp at (λ=3, κ=10%) | `run_cap_ablation.py` |
| Llama-3.1-8B: n paired | 143 | `reproduce_permutation.py` |
| Llama-3.1-8B: +3.50pp at deployed (λ=1, κ=5%) | +3.50pp, McNemar p=0.063 | `reproduce_permutation.py` |
| Llama peak governance grid | +7.7pp at (λ=2, κ=10%) | `run_cap_ablation.py` |
| 1000-perm scenario-clustered max-stat null | Qwen p<0.001, Llama p=0.008 | `reproduce_permutation.py` |
| 36-cell governance grid | Table 2 / Fig 1 | `run_cap_ablation.py` |
| Paper figures (Figs 1-3) | heatmap, attribution, trajectory | `generate_paper_figures.py` |

### Pairing key — critical for n=143 (paper L197 footnote)

`reproduce_permutation.py` uses the **3-tuple key** `(scenario_id, traveler_id, bundle_id)`
to match original vs. factual stimulus variants. This reproduces the paper's published
n=143 paired analysis. A 5-tuple key `(scenario_id, traveler_id, bundle_id, signal_wt,
episode_seed)` yields n=409 and does NOT match the paper's Table 1 or abstract.
The script explicitly documents this in its docstring and `--help` output.

### Permutation seed — paper-locked

The permutation null uses `--seed 12345`. **Changing this seed is a methodological
change that invalidates cross-paper comparison.** The `expected_outputs/` were
generated at this seed. The `verify.py` script checks that fresh runs produce
the same headline p-values (to within floating-point tolerance).

### Cluster unit

Cluster exchangeability is at the `scenario_id` level (88 unique scenarios for n=143 pairs),
matching paper §D / L294: "88 unique scenarios." Within each permutation, all pairs
sharing a scenario_id have their original↔factual labels flipped jointly.

---

## GPU re-generation (optional, ~94 min)

The raw.jsonl input files were generated on Server B (NVIDIA RTX 3090 24GB,
CUDA 12.1, PyTorch 2.4.0+cu121, vLLM 0.6+). To re-run from scratch:

| Stage | Script | Wall time |
|---|---|---|
| msgcap generation (Qwen-7B-Instruct, v4) | `run_phase1_msgcapture.py` | ~13 min |
| msgcap generation (Llama-3.1-8B, v6) | `run_phase1_msgcapture.py` | ~20 min |
| Six-gate audit | `run_phase1_judge.py` | <2 min |
| Phase 1c diagnostic paired replay | `run_phase1d_preflight.py` | ~51 min |
| Cap-ablation 36-cell grid | `run_cap_ablation.py` | ~8 min |

Total: ~94 min. All wall times are as stated in paper §7.2; camera-ready
re-benchmark with `torch.cuda.Event` instrumentation is a tracked obligation
(see `results/runtime_provenance.json`).

GPU scripts from `scripts/` are NOT copied here because they require a live
vLLM server and GPU environment. They are available in the full repository.

---

## Missing / pending (flagged)

- `MISSING: audit/regex_v2.patch` — explicit diff for six-gate refusal regex v1→v2. Camera-ready obligation. Documented in `results/gate_history.md §3`.
- `MISSING: audit_gates_pre_v5.yaml` — pre-v5 gate configuration snapshot. Camera-ready obligation.
- `MISSING: conda env manifest` — full Server B environment. Camera-ready obligation. See `results/runtime_provenance.json`.
(GitHub URL, LICENSE copyright, and CITATION.cff author block were resolved in the v1.0.0 submission.)

---

## Verification

```bash
# Verify input checksums + expected outputs match paper numbers:
python3 verify.py --results-dir ../../../results --expected-dir expected_outputs

# After running reproduce_all.sh, also verify fresh run outputs:
python3 verify.py \
    --results-dir ../../../results \
    --expected-dir expected_outputs \
    --run-outputs run_outputs/permutation_null
```

Expected: `ALL CHECKS PASSED.`

---

## Citation

See `CITATION.cff`. Cite: Liu, Y. (2026). *TourMart: A Parametric Audit Instrument for Commission Steering in LLM Travel Agents.* ORCID: [0009-0009-3128-7802](https://orcid.org/0009-0009-3128-7802).

## License

Code: MIT. Data (raw.jsonl, permutation outputs): CC-BY-4.0. See `LICENSE` and `DATA_CARD.md`.
