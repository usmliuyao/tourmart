#!/usr/bin/env bash
# Chain runner — post-v6 Llama OTA verbosity probe.
#
# Purpose: test whether Llama-3.1-8B's template-collapse msgcap failure (v6)
# can be repaired by appending VERBOSE_PROBE_SUFFIX to the OTA system prompt.
#
# Stage 1: launch vLLM (Llama-3.1-8B-Instruct) + run msgcap on 54 eps with
#          --prompt-variant verbose_probe, backbone_label=llama8b_probe_verbose.
# Stage 2: symmetric v7 audit — probe vs v6 archived (did suffix help?)
# Stage 3: symmetric v7 audit — probe vs Qwen-7B v4 (parity with default baseline?)
#
# Markers under ${RESULTS_DIR}/:
#   .probe_msgcap_started
#   .probe_msgcap_done       — set when msgcap jsonl has the expected episode count
#   .probe_audit_vs_v6_done  — audit improvement vs archived v6 complete
#   .probe_audit_vs_qwen_done — audit parity vs Qwen v4 baseline complete
#   .probe_complete          — all three stages passed cleanly
#   .probe_failed            — any stage failed

set -u
cd ${TOURMART_ROOT}

RESULTS=${TOURMART_ROOT}/results
LOG=$RESULTS/logs/phase1c_probe.log
mkdir -p "$RESULTS/logs"

LABEL="llama8b_probe_verbose"
EP_PATH="$RESULTS/phase1_msgcap_${LABEL}_episodes.jsonl"
MODEL=${MODELS_DIR_MS}/LLM-Research/Meta-Llama-3___1-8B-Instruct
AUDIT_VS_V6="$RESULTS/msgcap_v7_probe_vs_v6_audit.md"
AUDIT_VS_QWEN="$RESULTS/msgcap_v7_probe_vs_qwen_audit.md"
PROBE_V6_PASS="$RESULTS/.audit_probe_vs_v6_passed"
PROBE_V6_FAIL="$RESULTS/.audit_probe_vs_v6_failed"
PROBE_QWEN_PASS="$RESULTS/.audit_probe_vs_qwen_passed"
PROBE_QWEN_FAIL="$RESULTS/.audit_probe_vs_qwen_failed"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

rm -f "$RESULTS/.probe_msgcap_started" "$RESULTS/.probe_msgcap_done" \
      "$RESULTS/.probe_audit_vs_v6_done" "$RESULTS/.probe_audit_vs_qwen_done" \
      "$RESULTS/.probe_complete" "$RESULTS/.probe_failed" \
      "$PROBE_V6_PASS" "$PROBE_V6_FAIL" "$PROBE_QWEN_PASS" "$PROBE_QWEN_FAIL"

log "=== Phase1c Llama verbosity probe: chain start ==="

# ── Stage 1: msgcap with verbose_probe suffix ──────────────────────────────
log "Stage 1: launching msgcap (Llama-3.1-8B + verbose_probe, 54 eps target)"
touch "$RESULTS/.probe_msgcap_started"

timeout --kill-after=120s 2h \
  python scripts/run_phase1_msgcapture.py \
    --backend vllm \
    --model-path "$MODEL" \
    --tensor-parallel-size 1 \
    --backbone-label "$LABEL" \
    --per-condition 18 \
    --batch-size 32 \
    --sample-seed 42 \
    --priors-mode uniform \
    --prompt-variant verbose_probe \
  >> "$LOG" 2>&1
MSGCAP_RC=$?
log "msgcap exit code: $MSGCAP_RC"

if [[ $MSGCAP_RC -ne 0 ]]; then
  log "FATAL: msgcap failed (rc=$MSGCAP_RC); marking probe_failed"
  touch "$RESULTS/.probe_failed"
  exit 1
fi

# Confirm jsonl has enough episodes (54 target; accept ≥ 45).
EP_COUNT=$(wc -l < "$EP_PATH" 2>/dev/null | tr -d ' ')
log "msgcap jsonl episode count: $EP_COUNT"
if [[ -z "$EP_COUNT" || "$EP_COUNT" -lt 45 ]]; then
  log "FATAL: jsonl has < 45 episodes (got $EP_COUNT); marking probe_failed"
  touch "$RESULTS/.probe_failed"
  exit 1
fi
touch "$RESULTS/.probe_msgcap_done"

# ── Stage 2: symmetric v7 audit — probe vs archived v6 ────────────────────
log "Stage 2: symmetric audit — probe vs v6 archived (improvement check)"
PYTHONPATH=src python scripts/run_stimulus_audit_v7.py \
  --ota-jsonl "$EP_PATH" \
  --ota-label "Llama-3.1-8B probe (verbose)" \
  --baseline-jsonl "$RESULTS/archived_v6_llama_ota/phase1_msgcap_llama8b_msgcap_v6_episodes.jsonl" \
  --baseline-label "Llama-3.1-8B (v6, default prompt)" \
  --out "$AUDIT_VS_V6" \
  --pass-marker "$PROBE_V6_PASS" \
  --fail-marker "$PROBE_V6_FAIL" \
  >> "$LOG" 2>&1
AUDIT_V6_RC=$?
log "audit vs v6 exit code: $AUDIT_V6_RC"
touch "$RESULTS/.probe_audit_vs_v6_done"

# ── Stage 3: symmetric v7 audit — probe vs Qwen-7B v4 ─────────────────────
log "Stage 3: symmetric audit — probe vs Qwen-7B v4 (parity check)"
PYTHONPATH=src python scripts/run_stimulus_audit_v7.py \
  --ota-jsonl "$EP_PATH" \
  --ota-label "Llama-3.1-8B probe (verbose)" \
  --baseline-jsonl "$RESULTS/phase1_msgcap_qwen7b_msgcap_v4_episodes.jsonl" \
  --baseline-label "Qwen-7B (v4, default prompt)" \
  --out "$AUDIT_VS_QWEN" \
  --pass-marker "$PROBE_QWEN_PASS" \
  --fail-marker "$PROBE_QWEN_FAIL" \
  >> "$LOG" 2>&1
AUDIT_QWEN_RC=$?
log "audit vs Qwen v4 exit code: $AUDIT_QWEN_RC"
touch "$RESULTS/.probe_audit_vs_qwen_done"

# ── Final marker ──────────────────────────────────────────────────────────
# Probe is a DIAGNOSTIC. The primary gate is the v6-comparison: did the
# suffix meaningfully improve any gate that was failing in v6?
# Do NOT require probe to pass all OTA gates — that is the question, not a
# precondition. Any audit-script crash (rc != 0 AND rc != 1) is a real fail.
if [[ $AUDIT_V6_RC -gt 1 || $AUDIT_QWEN_RC -gt 1 ]]; then
  log "FATAL: audit script crashed (rc_v6=$AUDIT_V6_RC rc_qwen=$AUDIT_QWEN_RC)"
  touch "$RESULTS/.probe_failed"
  exit 1
fi

log "=== Probe chain complete. Inspect audit reports for verdict. ==="
log "  vs v6:   $AUDIT_VS_V6"
log "  vs Qwen: $AUDIT_VS_QWEN"
touch "$RESULTS/.probe_complete"
exit 0
