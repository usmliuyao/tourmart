#!/usr/bin/env bash
# =============================================================================
# TourMart — master reproduction runner
# Reproduces all paper headline numbers from pre-computed raw data files.
# No GPU required for Steps 1–4. GPU required only for Step 5 (LLM re-run).
# =============================================================================
set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve repo root (directory containing tourmart/paper and tourmart/results)
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_DIR="${SCRIPT_DIR}"
REPO_ROOT="${SCRIPT_DIR}/../../"             # tourmart/ parent
TOURMART_ROOT="$(realpath "${REPO_ROOT}/tourmart")"
RESULTS_DIR="${TOURMART_ROOT}/results"
SCRIPTS_DIR="${REPRO_DIR}/scripts"
EXPECTED_DIR="${REPRO_DIR}/expected_outputs"
OUT_DIR="${REPRO_DIR}/run_outputs"

export TOURMART_ROOT

mkdir -p "${OUT_DIR}"

echo "================================================================"
echo " TourMart Reproducibility Runner"
echo " TOURMART_ROOT : ${TOURMART_ROOT}"
echo " RESULTS_DIR   : ${RESULTS_DIR}"
echo " Output dir    : ${OUT_DIR}"
echo "================================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1: Verify input data checksums
# ---------------------------------------------------------------------------
echo "[Step 1] Verifying input data SHA-256 checksums..."
python3 "${REPRO_DIR}/verify.py" --check-inputs-only \
    --results-dir "${RESULTS_DIR}" \
    --expected-dir "${EXPECTED_DIR}"
echo "  PASS"
echo ""

# ---------------------------------------------------------------------------
# Step 2: Run cap-ablation (36-cell governance grid), 3-tuple SENSITIVITY.
#   Pure Python, no GPU. ~8 min on CPU. run_cap_ablation.py uses the legacy
#   3-tuple pairing, so its grid peaks (+10.5/+7.7pp) are the SENSITIVITY grid.
#   The PRIMARY 5-tuple grid peaks (+6.11/+10.02pp) and Fig 1 heatmap are
#   produced by Step 3a (reproduce_permutation_5tuple.py) and Step 4
#   (generate_paper_figures.py), and verified against expected_outputs.
# ---------------------------------------------------------------------------
echo "[Step 2] Cap-ablation 36-cell grid (3-tuple sensitivity)..."
python3 "${SCRIPTS_DIR}/run_cap_ablation.py" \
    --qwen-raw  "${RESULTS_DIR}/phase1c_qwen14b_awq_diag_v4_report.raw.jsonl" \
    --llama-raw "${RESULTS_DIR}/phase1c_llama31_8b_diag_v4_report.raw.jsonl" \
    --out "${OUT_DIR}/cap_ablation.md"
echo "  Wrote: ${OUT_DIR}/cap_ablation.md"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Permutation null (1000-perm scenario-clustered max-stat). Seed 12345.
#   PRIMARY = full-identity 5-tuple pairing (scenario, signal_wt, episode_seed,
#     traveler, bundle) -> n=409. Reproduces paper Table 1/3 + abstract:
#     Qwen grid peak +6.11pp p=0.001, Llama +10.02pp p=0.003.
#   SENSITIVITY = 3-tuple (scenario, traveler, bundle) -> n=143 (paper pairing-
#     convention block: Qwen +10.49pp, Llama +7.69pp). The 3-tuple last-write-
#     wins collapses distinct (signal_wt, episode_seed) episodes; the 5-tuple
#     does not, and is therefore primary.
#   Seed: 12345 (paper-locked; do not change).
# ---------------------------------------------------------------------------
echo "[Step 3a] PRIMARY 5-tuple permutation null (n=409)..."
python3 "${SCRIPTS_DIR}/reproduce_permutation_5tuple.py" \
    --qwen-raw  "${RESULTS_DIR}/phase1c_qwen14b_awq_diag_v4_report.with_episode_seed.raw.jsonl" \
    --llama-raw "${RESULTS_DIR}/phase1c_llama31_8b_diag_v4_report.with_episode_seed.raw.jsonl" \
    --out-dir   "${OUT_DIR}/permutation_null" \
    --n-perm    1000 \
    --seed      12345
echo "  Wrote: ${OUT_DIR}/permutation_null/permutation_summary_5tuple.json"
echo ""
echo "[Step 3b] SENSITIVITY 3-tuple permutation null (n=143)..."
python3 "${SCRIPTS_DIR}/reproduce_permutation.py" \
    --qwen-raw  "${RESULTS_DIR}/phase1c_qwen14b_awq_diag_v4_report.with_episode_seed.raw.jsonl" \
    --llama-raw "${RESULTS_DIR}/phase1c_llama31_8b_diag_v4_report.with_episode_seed.raw.jsonl" \
    --out-dir   "${OUT_DIR}/permutation_null" \
    --n-perm    1000 \
    --seed      12345
echo "  Wrote: ${OUT_DIR}/permutation_null/permutation_summary.json"
echo ""
echo "[Step 3c] Deployed-point cluster-robust inference (Table 1 clustered p)..."
python3 "${SCRIPTS_DIR}/deployed_clustered_inference.py" \
    "${RESULTS_DIR}/phase1c_qwen14b_awq_diag_v4_report.with_episode_seed.raw.jsonl" \
    "${RESULTS_DIR}/phase1c_llama31_8b_diag_v4_report.with_episode_seed.raw.jsonl"
echo ""

# ---------------------------------------------------------------------------
# Step 4: Generate paper figures (Figs 1-3)
#   Pure Python + matplotlib, no GPU.
# ---------------------------------------------------------------------------
echo "[Step 4] Generating paper figures..."
python3 "${SCRIPTS_DIR}/generate_paper_figures.py"
echo "  Wrote: ${REPRO_DIR}/figures/"
echo ""

# ---------------------------------------------------------------------------
# Step 5: Verify outputs against expected
# ---------------------------------------------------------------------------
echo "[Step 5] Comparing run outputs against expected_outputs/..."
python3 "${REPRO_DIR}/verify.py" \
    --results-dir "${RESULTS_DIR}" \
    --expected-dir "${EXPECTED_DIR}" \
    --run-outputs "${OUT_DIR}/permutation_null"
echo ""

echo "================================================================"
echo " All steps complete."
echo " Review: ${OUT_DIR}/"
echo "================================================================"
echo ""
echo "NOTE — GPU re-generation (not run by this script):"
echo "  To re-generate the raw.jsonl files from scratch (requires"
echo "  NVIDIA RTX 3090 + CUDA 12.1 + vLLM 0.6+, ~94 min total):"
echo "    Server B: ssh <your-gpu-server>"
echo "    See runtime_provenance.json for stage-by-stage wall times."
