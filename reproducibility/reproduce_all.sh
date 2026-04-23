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
# Step 2: Run cap-ablation (36-cell governance grid)
#   Pure Python, no GPU. ~8 min on CPU.
#   Reproduces Table 2 / Fig 1 RD grid numbers.
# ---------------------------------------------------------------------------
echo "[Step 2] Cap-ablation 36-cell grid..."
python3 "${SCRIPTS_DIR}/run_cap_ablation.py" \
    --qwen-raw  "${RESULTS_DIR}/phase1c_qwen14b_awq_diag_v4_report.raw.jsonl" \
    --llama-raw "${RESULTS_DIR}/phase1c_llama31_8b_diag_v4_report.raw.jsonl" \
    --out "${OUT_DIR}/cap_ablation.md"
echo "  Wrote: ${OUT_DIR}/cap_ablation.md"
echo ""

# ---------------------------------------------------------------------------
# Step 3: Run permutation null (1000-perm scenario-clustered max-stat)
#   Pure Python + numpy, no GPU. ~2-5 min on CPU.
#   Reproduces: Qwen p<0.001, Llama p=0.001 (paper abstract; paper §D/L294).
#   Pairing: 3-tuple (scenario_id, traveler_id, bundle_id) → n=143 pairs.
#   Seed: 12345 (paper-locked; do not change).
# ---------------------------------------------------------------------------
echo "[Step 3] 1000-permutation scenario-clustered max-stat null..."
python3 "${SCRIPTS_DIR}/reproduce_permutation.py" \
    --qwen-raw  "${RESULTS_DIR}/phase1c_qwen14b_awq_diag_v4_report.with_episode_seed.raw.jsonl" \
    --llama-raw "${RESULTS_DIR}/phase1c_llama31_8b_diag_v4_report.with_episode_seed.raw.jsonl" \
    --out-dir   "${OUT_DIR}/permutation_null" \
    --n-perm    1000 \
    --seed      12345
echo "  Wrote: ${OUT_DIR}/permutation_null/"
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
