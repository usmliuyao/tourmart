# TourMart

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19709369.svg)](https://doi.org/10.5281/zenodo.19709369)

A parametric audit instrument for commission steering in LLM travel agents.

**Author**: [Yao Liu](https://orcid.org/0009-0009-3128-7802)
· Chengdu University of Technology · Universiti Sains Malaysia
**License**: MIT (code); CC-BY-4.0 (generated data and audit outputs)

---

## What this repository contains

1. A multi-agent market simulator with a welfare rule governed by two
   parameters: `lambda` (weight of message-induced perceptions on the accept
   decision) and `kappa` (cap on net message-induced welfare adjustment,
   budget-normalized).
2. The Qwen-7B OTA message-capture pipeline (v4 frozen) that produces
   counterfactual commission-maximizing vs. satisfaction-maximizing prose.
3. The Phase 1c paired-replay pipeline for two traveler-reader backbones:
   Qwen2.5-14B-Instruct-AWQ and Llama-3.1-8B-Instruct.
4. The 36-cell `(lambda, kappa)` governance-grid sweep.
5. The symmetric six-gate audit and the regex v2 refusal classifier.
6. All 143 paired raw traveler outputs with extracted perception features.
7. The four audit markdown reports cited in the paper's Appendix D.3.

The `reproducibility/` package reproduces Table 1, Table 2, and
Figures 1-3 in under 10 minutes on CPU.

---

## Quick start — CPU-only reproduction (no GPU required)

```bash
git clone https://github.com/usmliuyao/tourmart
cd tourmart
pip install 'numpy>=2.0' 'scipy>=1.13' 'statsmodels>=0.14' 'matplotlib>=3.8' 'pandas>=2.2'

# Verify raw-data checksums and confirm headline numbers
python3 reproducibility/verify.py \
    --results-dir results/ \
    --expected-dir reproducibility/expected_outputs/

# Run the full reproduction pipeline
bash reproducibility/reproduce_all.sh
```

Expected output: `ALL CHECKS PASSED`, `+10.49pp` peak grid (Qwen-14B-AWQ arm),
`+7.69pp` peak grid (Llama-3.1-8B arm), matched to within rounding tolerance.

---

## GPU regeneration (optional, ~94 min on one RTX 3090 24GB)

To re-run the full message-capture + replay pipeline from scratch you need a
CUDA-12.1-capable NVIDIA GPU with 24 GB or more of VRAM. Model downloads are
roughly 50 GB total (Qwen-7B-Instruct, Qwen-14B-Instruct-AWQ,
Meta-Llama-3.1-8B-Instruct).

1. Edit `scripts/run_config.sh.template`, save as `scripts/run_config.sh`,
   adjust `MODELS_DIR` etc. for your paths.
2. `source scripts/run_config.sh`
3. `bash scripts/chain_runner_phase1c_v4.sh`

Runtime budget: ~13 min (msgcap) + ~51 min (Phase 1c replay) + ~8 min
(cap-ablation grid).

---

## Repository layout

```
tourmart/
├── src/tourmart/                      Python package: market, audit, stats
├── scripts/
│   ├── run_config.sh.template           copy -> run_config.sh, source before chain_runners
│   ├── chain_runner_phase1c_*.sh        end-to-end pipelines (GPU)
│   ├── run_phase1_msgcapture.py         OTA msgcap (Qwen-7B)
│   ├── run_phase1c_crossfamily.py       paired replay (Qwen-14B + Llama-8B)
│   ├── run_cap_ablation.py              36-cell governance grid
│   ├── run_stimulus_audit_v6.py         six-gate audit
│   ├── run_stimulus_audit_v7.py         symmetric cross-arm audit
│   ├── run_coefficient_attribution.py   feature attribution
│   ├── reproduce_permutation.py         cluster-bootstrap permutation null
│   └── generate_paper_figures.py        Figs 1-3
├── results/                           143 paired raw outputs + audit markdowns
├── tests/                             unit tests for market, stats, audit
├── reproducibility/                   self-contained CPU-only repro package
├── figures/                           Figs 1-3 as PDF
├── LICENSE                              MIT (code) + CC-BY-4.0 (data)
├── CITATION.cff                         citation metadata
└── pyproject.toml
```

---

## Citation

If you use TourMart in your research, please cite:

```
@software{liu_tourmart_2026,
  author  = {Liu, Yao},
  title   = {TourMart: A Parametric Audit Instrument for Commission Steering in LLM Travel Agents},
  year    = {2026},
  version = {1.0.0},
  doi     = {10.5281/zenodo.19709369},
  url     = {https://doi.org/10.5281/zenodo.19709369}
}
```

See `CITATION.cff` for the canonical metadata.
