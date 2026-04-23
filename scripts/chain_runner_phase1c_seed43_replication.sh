#!/usr/bin/env bash
# Chain runner — seed 43 replication + batch=128 sanity check for §5.3.
#
# Spec (frozen by Codex battle Round 2, 2026-04-20):
#   Stage 1: Llama default-prompt @ seed 43, per_condition=150 (150 commission eps).
#            Label: llama8b_default_seed43.
#            Purpose: cross-seed replication of Claim 1 (template collapse).
#   Stage 2: Llama verbose_probe @ seed 43, per_condition=18 (54 commission eps).
#            Label: llama8b_probe_verbose_seed43.
#            Purpose: cross-seed replication of Claim 2 (prompt-fix works).
#   Stage 3: Llama verbose_probe @ seed 42, batch_size=128, per_condition=18.
#            Label: llama8b_probe_verbose_batch128.
#            Purpose: removes batch-size confound between v6 (batch 128) and
#            original probe (batch 32).
#   Stage 4: 4 audit v7 runs with the expanded refusal regex.
#            (classifier was hardened post-battle to include "unfortunately",
#            "couldn't find", etc.)
#
# Markers under ${RESULTS_DIR}/:
#   .seed43_stage1_done    — default seed 43 msgcap complete
#   .seed43_stage2_done    — probe seed 43 msgcap complete
#   .seed43_stage3_done    — probe batch=128 msgcap complete
#   .seed43_audits_done    — all 4 audits complete
#   .seed43_complete       — all 4 stages succeeded
#   .seed43_failed         — any stage failed

set -u
cd ${TOURMART_ROOT}

RESULTS=${TOURMART_ROOT}/results
LOG=$RESULTS/logs/phase1c_seed43.log
mkdir -p "$RESULTS/logs"

MODEL=${MODELS_DIR_MS}/LLM-Research/Meta-Llama-3___1-8B-Instruct

L1=llama8b_default_seed43
L2=llama8b_probe_verbose_seed43
L3=llama8b_probe_verbose_batch128
EP1="$RESULTS/phase1_msgcap_${L1}_episodes.jsonl"
EP2="$RESULTS/phase1_msgcap_${L2}_episodes.jsonl"
EP3="$RESULTS/phase1_msgcap_${L3}_episodes.jsonl"

V6_JSONL="$RESULTS/archived_v6_llama_ota/phase1_msgcap_llama8b_msgcap_v6_episodes.jsonl"
PROBE42_JSONL="$RESULTS/phase1_msgcap_llama8b_probe_verbose_episodes.jsonl"
QWEN_V4_JSONL="$RESULTS/phase1_msgcap_qwen7b_msgcap_v4_episodes.jsonl"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

rm -f "$RESULTS"/.seed43_stage*_done "$RESULTS"/.seed43_audits_done \
      "$RESULTS"/.seed43_complete "$RESULTS"/.seed43_failed

log "=== seed43 replication chain start ==="

# ── Stage 1: Llama default seed 43, 150 commission eps ─────────────────────
log "Stage 1: Llama default-prompt @ seed 43 (per_cond=150, batch=32)"
timeout --kill-after=120s 2h \
  python scripts/run_phase1_msgcapture.py \
    --backend vllm \
    --model-path "$MODEL" \
    --tensor-parallel-size 1 \
    --backbone-label "$L1" \
    --per-condition 150 \
    --batch-size 32 \
    --sample-seed 43 \
    --priors-mode uniform \
    --prompt-variant default \
  >> "$LOG" 2>&1
RC1=$?
log "Stage 1 exit: $RC1"
if [[ $RC1 -ne 0 ]]; then
  touch "$RESULTS/.seed43_failed"; exit 1
fi
EP1_COUNT=$(wc -l < "$EP1" 2>/dev/null | tr -d ' ')
log "Stage 1 jsonl count: $EP1_COUNT"
if [[ -z "$EP1_COUNT" || "$EP1_COUNT" -lt 400 ]]; then
  log "FATAL: Stage 1 jsonl < 400 eps (got $EP1_COUNT)"
  touch "$RESULTS/.seed43_failed"; exit 1
fi
touch "$RESULTS/.seed43_stage1_done"

# ── Stage 2: Llama probe seed 43, 54 commission eps ────────────────────────
log "Stage 2: Llama verbose_probe @ seed 43 (per_cond=18, batch=32)"
timeout --kill-after=120s 1h \
  python scripts/run_phase1_msgcapture.py \
    --backend vllm \
    --model-path "$MODEL" \
    --tensor-parallel-size 1 \
    --backbone-label "$L2" \
    --per-condition 18 \
    --batch-size 32 \
    --sample-seed 43 \
    --priors-mode uniform \
    --prompt-variant verbose_probe \
  >> "$LOG" 2>&1
RC2=$?
log "Stage 2 exit: $RC2"
if [[ $RC2 -ne 0 ]]; then
  touch "$RESULTS/.seed43_failed"; exit 1
fi
EP2_COUNT=$(wc -l < "$EP2" 2>/dev/null | tr -d ' ')
log "Stage 2 jsonl count: $EP2_COUNT"
if [[ -z "$EP2_COUNT" || "$EP2_COUNT" -lt 45 ]]; then
  log "FATAL: Stage 2 jsonl < 45 eps (got $EP2_COUNT)"
  touch "$RESULTS/.seed43_failed"; exit 1
fi
touch "$RESULTS/.seed43_stage2_done"

# ── Stage 3: probe seed 42 @ batch=128 (batch-size confound check) ─────────
log "Stage 3: Llama verbose_probe @ seed 42 batch=128 (per_cond=18)"
timeout --kill-after=120s 1h \
  python scripts/run_phase1_msgcapture.py \
    --backend vllm \
    --model-path "$MODEL" \
    --tensor-parallel-size 1 \
    --backbone-label "$L3" \
    --per-condition 18 \
    --batch-size 128 \
    --sample-seed 42 \
    --priors-mode uniform \
    --prompt-variant verbose_probe \
  >> "$LOG" 2>&1
RC3=$?
log "Stage 3 exit: $RC3"
if [[ $RC3 -ne 0 ]]; then
  touch "$RESULTS/.seed43_failed"; exit 1
fi
EP3_COUNT=$(wc -l < "$EP3" 2>/dev/null | tr -d ' ')
log "Stage 3 jsonl count: $EP3_COUNT"
if [[ -z "$EP3_COUNT" || "$EP3_COUNT" -lt 45 ]]; then
  log "FATAL: Stage 3 jsonl < 45 eps (got $EP3_COUNT)"
  touch "$RESULTS/.seed43_failed"; exit 1
fi
touch "$RESULTS/.seed43_stage3_done"

# ── Stage 4: audits (with expanded refusal regex from post-battle classifier) ──
log "Stage 4a: audit — seed43_default vs v6 archived (cross-seed Claim 1)"
PYTHONPATH=src python scripts/run_stimulus_audit_v7.py \
  --ota-jsonl "$EP1" --ota-label "Llama seed43 default" \
  --baseline-jsonl "$V6_JSONL" --baseline-label "Llama v6 default (seed42)" \
  --out "$RESULTS/msgcap_v7_seed43_default_vs_v6.md" \
  --pass-marker "$RESULTS/.audit_seed43_default_vs_v6_passed" \
  --fail-marker "$RESULTS/.audit_seed43_default_vs_v6_failed" \
  >> "$LOG" 2>&1 || true

log "Stage 4b: audit — seed43_probe vs seed43_default (within-seed Claim 2)"
PYTHONPATH=src python scripts/run_stimulus_audit_v7.py \
  --ota-jsonl "$EP2" --ota-label "Llama seed43 probe (verbose)" \
  --baseline-jsonl "$EP1" --baseline-label "Llama seed43 default" \
  --out "$RESULTS/msgcap_v7_seed43_probe_vs_seed43_default.md" \
  --pass-marker "$RESULTS/.audit_seed43_probe_vs_default_passed" \
  --fail-marker "$RESULTS/.audit_seed43_probe_vs_default_failed" \
  >> "$LOG" 2>&1 || true

log "Stage 4c: audit — batch128 probe vs batch32 probe (seed 42, batch confound)"
PYTHONPATH=src python scripts/run_stimulus_audit_v7.py \
  --ota-jsonl "$EP3" --ota-label "Llama seed42 probe batch=128" \
  --baseline-jsonl "$PROBE42_JSONL" --baseline-label "Llama seed42 probe batch=32" \
  --out "$RESULTS/msgcap_v7_batch128_vs_batch32.md" \
  --pass-marker "$RESULTS/.audit_batch_confound_passed" \
  --fail-marker "$RESULTS/.audit_batch_confound_failed" \
  >> "$LOG" 2>&1 || true

log "Stage 4d: re-audit — v6 vs Qwen v4 with expanded refusal regex"
PYTHONPATH=src python scripts/run_stimulus_audit_v7.py \
  --ota-jsonl "$V6_JSONL" --ota-label "Llama v6 (regex-v2)" \
  --baseline-jsonl "$QWEN_V4_JSONL" --baseline-label "Qwen v4 (regex-v2)" \
  --out "$RESULTS/msgcap_v7_v6_vs_qwen_regex_v2.md" \
  --pass-marker "$RESULTS/.audit_v6_vs_qwen_v2_passed" \
  --fail-marker "$RESULTS/.audit_v6_vs_qwen_v2_failed" \
  >> "$LOG" 2>&1 || true

touch "$RESULTS/.seed43_audits_done"
log "=== seed43 replication chain complete ==="
touch "$RESULTS/.seed43_complete"
exit 0
