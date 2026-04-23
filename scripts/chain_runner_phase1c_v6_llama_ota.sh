#!/bin/bash
# Chain runner v6 — Llama-3.1-8B-Instruct OTA cross-family robustness.
# Pre-reg: refine-logs/PHASE1C_V6_LLAMA_OTA_PREREG.md (frozen 2026-04-20).
#
# Changes vs v5 (GPT-5.4 adversarial review, 2026-04-20):
#   - Clean stale markers at entry (preserve .chain_v6_started on rerun).
#   - msgcap runs in foreground inside outer chain screen (no nested screen).
#   - GNU timeout on every long step with distinct *_timeout markers.
#   - Audit step uses direct exit-code check AND pass-marker assertion
#     (does not trust absence of fail marker alone).
#   - Post-msgcap integrity gate: min line count + JSON parse sanity check.
#   - Sanity check all 3 model paths upfront.
#   - Batch size 64 (down from 128) for Llama-8B bf16 on 24GB 3090.
#   - trap writes .chain_v6_failed_shell_trap on unexpected exit.
#
# Launch:
#   ssh <your-gpu-server> \
#     "screen -dmS chain_p1c_v6 bash ${TOURMART_ROOT}/scripts/chain_runner_phase1c_v6_llama_ota.sh"
set -euo pipefail
cd ${TOURMART_ROOT}

LOG=${LOGS_DIR}/chain_runner_v6.log
mkdir -p ${TOURMART_ROOT}/logs ${TOURMART_ROOT}/results
exec > "$LOG" 2>&1

MSGCAP_JSONL=${RESULTS_DIR}/phase1_msgcap_llama8b_msgcap_v6_episodes.jsonl
QWEN_V4_JSONL=${RESULTS_DIR}/phase1_msgcap_qwen7b_msgcap_v4_episodes.jsonl
LLAMA_PATH=${MODELS_DIR_MS}/LLM-Research/Meta-Llama-3___1-8B-Instruct
QWEN14B_AWQ_PATH=${MODELS_DIR}/Qwen/Qwen2.5-14B-Instruct-AWQ
M=${TOURMART_ROOT}/results

log() { echo "[$(date '+%F %T')] $*"; }

# ========== trap: catch unexpected shell exits ==========
trap 'rc=$?; if [ $rc -ne 0 ] && [ ! -f "$M/.chain_v6_all_done" ]; then
        touch "$M/.chain_v6_failed_shell_trap"
        log "TRAP: chain exited rc=$rc without completion marker."
      fi' EXIT

# ========== Step -1: Clean stale markers (keep a run diary) ==========
log "=== chain v6 Llama-OTA starting ==="
for f in .chain_v6_msgcap_ok .chain_v6_audit_ok .chain_v6_phase1c_qwen_ok \
         .chain_v6_phase1c_llama_ok .chain_v6_all_done \
         .chain_v6_failed_model_missing .chain_v6_failed_baseline_missing \
         .chain_v6_failed_msgcap .chain_v6_failed_msgcap_timeout \
         .chain_v6_failed_msgcap_integrity .chain_v6_failed_audit \
         .chain_v6_failed_audit_marker_missing .chain_v6_failed_qwen \
         .chain_v6_failed_qwen_timeout .chain_v6_failed_llama \
         .chain_v6_failed_llama_timeout .chain_v6_failed_shell_trap \
         .audit_v6_passed .audit_v6_failed; do
  rm -f "$M/$f"
done
touch "$M/.chain_v6_started"
log "cleaned stale markers; wrote .chain_v6_started"

# ========== Step 0: Sanity check all 3 model paths + baseline ==========
for p in "$LLAMA_PATH/config.json" "$QWEN14B_AWQ_PATH/config.json"; do
  if [ ! -f "$p" ]; then
    log "FAIL: missing required model file $p"
    touch "$M/.chain_v6_failed_model_missing"
    exit 1
  fi
done
log "model paths OK: $LLAMA_PATH, $QWEN14B_AWQ_PATH"

if [ ! -s "$QWEN_V4_JSONL" ]; then
  log "FAIL: v4 Qwen-OTA baseline jsonl missing at $QWEN_V4_JSONL (needed for audit comparison)."
  touch "$M/.chain_v6_failed_baseline_missing"
  exit 1
fi
log "v4 Qwen-OTA baseline present: $(wc -l < "$QWEN_V4_JSONL") lines."

# Optional preflight info
if command -v nvidia-smi >/dev/null 2>&1; then
  log "--- nvidia-smi snapshot ---"
  nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv | tee -a "$LOG" || true
fi
log "--- disk snapshot ---"
df -h /hy-tmp ${TOURMART_ROOT} 2>/dev/null | tee -a "$LOG" || true

# ========== Step 1: msgcap v6 with Llama OTA — foreground, timeout 90min ==========
# Clean any stale partial output before a fresh run.
if [ -f "$MSGCAP_JSONL" ] && [ ! -s "$MSGCAP_JSONL" ]; then
  log "stale empty msgcap file found, removing."
  rm -f "$MSGCAP_JSONL"
fi

if [ ! -s "$MSGCAP_JSONL" ]; then
  log "Launching msgcap v6 (Llama OTA) in foreground with 90min timeout..."
  set +e
  timeout --kill-after=120s 90m env PYTHONPATH=${TOURMART_ROOT}/src \
    python ${TOURMART_ROOT}/scripts/run_phase1_msgcapture.py \
      --backend vllm \
      --model-path "$LLAMA_PATH" \
      --backbone-label llama8b_msgcap_v6 \
      --per-condition 4500 \
      --batch-size 64 \
      > ${LOGS_DIR}/msgcap_v6.log 2>&1
  rc=$?
  set -e
  if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
    log "FAIL: msgcap v6 timed out (rc=$rc, 90min)."
    touch "$M/.chain_v6_failed_msgcap_timeout"
    exit 1
  elif [ $rc -ne 0 ]; then
    log "FAIL: msgcap v6 exited rc=$rc"
    touch "$M/.chain_v6_failed_msgcap"
    exit 1
  fi
  log "msgcap v6 exited rc=0."
else
  log "msgcap v6 jsonl already present; skipping regeneration (resume mode)."
fi

# ========== Step 1.5: Post-msgcap integrity check ==========
if [ ! -s "$MSGCAP_JSONL" ]; then
  log "FAIL: $MSGCAP_JSONL missing/empty after msgcap v6."
  touch "$M/.chain_v6_failed_msgcap"
  exit 1
fi

LINES=$(wc -l < "$MSGCAP_JSONL")
log "msgcap output lines: $LINES"
if [ "$LINES" -lt 200 ]; then
  log "FAIL: msgcap produced only $LINES lines (expected >=200 based on v4)."
  touch "$M/.chain_v6_failed_msgcap_integrity"
  exit 1
fi

# JSON parse sanity: first 5 lines, last 5 lines, must all parse.
set +e
python - <<PYEOF > /tmp/v6_integrity.txt 2>&1
import json, sys, itertools
p = "$MSGCAP_JSONL"
bad = 0
total = 0
with open(p) as f:
    head = list(itertools.islice(f, 5))
with open(p) as f:
    tail = []
    for line in f:
        tail.append(line)
        if len(tail) > 5:
            tail.pop(0)
for line in head + tail:
    total += 1
    try:
        json.loads(line)
    except Exception as e:
        bad += 1
        print(f"BAD: {e}")
print(f"sampled={total} bad={bad}")
sys.exit(0 if bad == 0 else 2)
PYEOF
rc=$?
set -e
log "integrity sample parse rc=$rc:"
cat /tmp/v6_integrity.txt | tee -a "$LOG"
if [ $rc -ne 0 ]; then
  log "FAIL: sampled msgcap lines did not all parse as JSON."
  touch "$M/.chain_v6_failed_msgcap_integrity"
  exit 1
fi

touch "$M/.chain_v6_msgcap_ok"
log "msgcap integrity OK."

# ========== Step 2: Stimulus-quality audit — direct exit code + pass-marker ==========
log "Running stimulus audit v6..."
set +e
timeout --kill-after=30s 10m env PYTHONPATH=${TOURMART_ROOT}/src \
  python ${TOURMART_ROOT}/scripts/run_stimulus_audit_v6.py \
    --ota-jsonl "$MSGCAP_JSONL" \
    --ota-label "Llama-3.1-8B (v6)" \
    --baseline-jsonl "$QWEN_V4_JSONL" \
    --baseline-label "Qwen2.5-7B (v4 baseline)" \
    --out "$M/msgcap_v6_stimulus_audit.md" \
    --pass-marker "$M/.audit_v6_passed" \
    --fail-marker "$M/.audit_v6_failed"
audit_rc=$?
set -e

if [ $audit_rc -ne 0 ]; then
  log "Audit exited rc=$audit_rc (means: at least one pre-reg gate failed OR script error)."
  touch "$M/.chain_v6_failed_audit"
  log "Audit report tail:"
  tail -25 "$M/msgcap_v6_stimulus_audit.md" 2>/dev/null || log "(no audit markdown file)"
  exit 1
fi

# Belt and suspenders: require pass marker to exist.
if [ ! -f "$M/.audit_v6_passed" ]; then
  log "FAIL: audit exited rc=0 but .audit_v6_passed not present."
  touch "$M/.chain_v6_failed_audit_marker_missing"
  exit 1
fi

log "Audit PASSED."
touch "$M/.chain_v6_audit_ok"

# ========== Step 3: Phase 1c Qwen-14B-AWQ (Llama stimuli) — PRIMARY ==========
log "Phase 1c v6 Qwen-14B-AWQ traveler arm (PRIMARY off-diagonal cell, seed=42)..."
set +e
timeout --kill-after=120s 120m env PYTHONPATH=${TOURMART_ROOT}/src \
  python ${TOURMART_ROOT}/scripts/run_phase1c_crossfamily.py \
    --msgcap-jsonl "$MSGCAP_JSONL" \
    --model-path "$QWEN14B_AWQ_PATH" \
    --arm-label qwen14b_awq_diag_v6 \
    --quantization awq_marlin \
    --dtype float16 \
    --diagnostic-window \
    --per-stratum 100 \
    --seed 42 \
    > ${LOGS_DIR}/p1c_v6_qwen.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
  log "FAIL: Phase 1c v6 Qwen timed out (rc=$rc, 120min)."
  touch "$M/.chain_v6_failed_qwen_timeout"
  exit 1
elif [ $rc -ne 0 ]; then
  log "FAIL: Phase 1c v6 Qwen rc=$rc"
  touch "$M/.chain_v6_failed_qwen"
  exit 1
fi
log "Phase 1c v6 Qwen DONE."
touch "$M/.chain_v6_phase1c_qwen_ok"
sleep 30

# ========== Step 4: Phase 1c Llama-3.1-8B — SECONDARY (diagonal) ==========
log "Phase 1c v6 Llama-3.1-8B traveler arm (SECONDARY diagonal cell, seed=42)..."
set +e
timeout --kill-after=120s 120m env PYTHONPATH=${TOURMART_ROOT}/src \
  python ${TOURMART_ROOT}/scripts/run_phase1c_crossfamily.py \
    --msgcap-jsonl "$MSGCAP_JSONL" \
    --model-path "$LLAMA_PATH" \
    --arm-label llama31_8b_diag_v6 \
    --dtype bfloat16 \
    --diagnostic-window \
    --per-stratum 100 \
    --seed 42 \
    > ${LOGS_DIR}/p1c_v6_llama.log 2>&1
rc=$?
set -e
if [ $rc -eq 124 ] || [ $rc -eq 137 ]; then
  log "FAIL: Phase 1c v6 Llama timed out (rc=$rc, 120min)."
  touch "$M/.chain_v6_failed_llama_timeout"
  exit 1
elif [ $rc -ne 0 ]; then
  log "FAIL: Phase 1c v6 Llama rc=$rc"
  touch "$M/.chain_v6_failed_llama"
  exit 1
fi
log "Phase 1c v6 Llama DONE."
touch "$M/.chain_v6_phase1c_llama_ok"

# ========== Final ==========
log "CHAIN v6 COMPLETE. Artifacts:"
ls -lht "$M"/phase1c_*diag_v6* 2>/dev/null | tee -a "$LOG"
ls -lht "$M"/phase1_msgcap_llama8b_* "$M"/msgcap_v6_* 2>/dev/null | tee -a "$LOG"
touch "$M/.chain_v6_all_done"
log "Wrote .chain_v6_all_done."
