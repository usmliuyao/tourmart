#!/bin/bash
# Chain runner v4 — durable scale-up to ~144 unique near-threshold pairs.
#
# Differences from v2:
#   - Scenarios expanded 25→75 per regime (3× pool)
#   - msgcap backbone label: qwen7b_msgcap_v4
#   - Phase 1c arm labels: qwen14b_awq_diag_v4, llama31_8b_diag_v4
#   - --per-stratum 100 (from 50) to fully sample the larger pool
#
# Launch:
#   ssh <your-gpu-server> \
#     "screen -dmS chain_p1c_v4 bash ${TOURMART_ROOT}/scripts/chain_runner_phase1c_v4.sh"
set -euo pipefail
cd ${TOURMART_ROOT}

LOG=${LOGS_DIR}/chain_runner_v4.log
mkdir -p ${TOURMART_ROOT}/logs ${TOURMART_ROOT}/results
exec > "$LOG" 2>&1

MSGCAP_JSONL=${RESULTS_DIR}/phase1_msgcap_qwen7b_msgcap_v4_episodes.jsonl
M=${TOURMART_ROOT}/results
touch "$M/.chain_v4_started"

log() { echo "[$(date '+%F %T')] $*"; }

# ========== Step 1: Launch msgcap v4 as background screen, then wait ==========
log "Launching msgcap v4 in screen msgcap_v4..."
if ! screen -list 2>/dev/null | grep -q 'msgcap_v4'; then
  screen -dmS msgcap_v4 bash -lc '
    cd ${TOURMART_ROOT} && PYTHONPATH=${TOURMART_ROOT}/src python scripts/run_phase1_msgcapture.py \
      --backend vllm \
      --model-path ${MODELS_DIR}/Qwen/Qwen2.5-7B-Instruct \
      --backbone-label qwen7b_msgcap_v4 \
      --per-condition 4500 \
      --batch-size 128 \
      > ${LOGS_DIR}/msgcap_v4.log 2>&1
  '
  sleep 3
fi

log "Waiting for msgcap_v4 screen to finish..."
WAITED=0
MAX_WAIT=10800  # 3h ceiling
while screen -list 2>/dev/null | grep -q 'msgcap_v4'; do
  sleep 30
  WAITED=$((WAITED + 30))
  if [ "$WAITED" -ge "$MAX_WAIT" ]; then
    log "FAIL: msgcap_v4 exceeded ${MAX_WAIT}s."
    touch "$M/.chain_v4_failed_msgcap_timeout"
    exit 1
  fi
done
log "msgcap_v4 exited after ${WAITED}s."

if [ ! -s "$MSGCAP_JSONL" ]; then
  log "FAIL: $MSGCAP_JSONL missing/empty."
  touch "$M/.chain_v4_failed_jsonl_missing"
  exit 1
fi

# ========== Step 1.5: Sanity check ==========
log "Sanity check..."
python3 - <<'PYEOF' > "$M/msgcap_v4_sanity.txt" 2>&1
import json
total = malformed = commission = valid_commission = 0
for line in open('${RESULTS_DIR}/phase1_msgcap_qwen7b_msgcap_v4_episodes.jsonl'):
    d = json.loads(line); total += 1
    if d.get('final_malformed'): malformed += 1
    if d.get('condition') == 'commission':
        commission += 1
        if not d.get('final_malformed'): valid_commission += 1
pct = int(malformed / max(total, 1) * 100)
print(f'total={total}')
print(f'malformed={malformed}  ({pct}%)')
print(f'commission={commission}')
print(f'valid_commission={valid_commission}')
with open('${RESULTS_DIR}/.msgcap_v4_malformed_pct', 'w') as f:
    f.write(str(pct))
with open('${RESULTS_DIR}/.msgcap_v4_valid_commission', 'w') as f:
    f.write(str(valid_commission))
PYEOF
cat "$M/msgcap_v4_sanity.txt"

MALFORMED_PCT=$(cat "$M/.msgcap_v4_malformed_pct")
VALID_COMM=$(cat "$M/.msgcap_v4_valid_commission")

if [ "$MALFORMED_PCT" -gt 15 ]; then
  log "FAIL: malformed ${MALFORMED_PCT}% > 15%. Aborting."
  touch "$M/.chain_v4_failed_malformed"
  exit 1
fi
if [ "$VALID_COMM" -lt 1000 ]; then
  log "WARN: valid_commission=${VALID_COMM} < 1000. May be underpowered."
  touch "$M/.chain_v4_warning_low_valid_commission"
fi
log "Malformed=${MALFORMED_PCT}%, valid_commission=${VALID_COMM}. OK."
touch "$M/.chain_v4_msgcap_ok"

# ========== Step 2: Phase 1c Qwen v4 ==========
log "Phase 1c Qwen-14B-AWQ v4 (diagnostic, seed=42, per-stratum=100)..."
PYTHONPATH=${TOURMART_ROOT}/src python ${TOURMART_ROOT}/scripts/run_phase1c_crossfamily.py \
    --msgcap-jsonl "$MSGCAP_JSONL" \
    --model-path ${MODELS_DIR}/Qwen/Qwen2.5-14B-Instruct-AWQ \
    --arm-label qwen14b_awq_diag_v4 \
    --quantization awq_marlin \
    --dtype float16 \
    --diagnostic-window \
    --per-stratum 100 \
    --seed 42 \
    > ${LOGS_DIR}/p1c_v4_qwen.log 2>&1 || {
        log "FAIL: Phase 1c Qwen v4 exited non-zero"
        touch "$M/.chain_v4_failed_qwen"
        exit 1
    }
log "Phase 1c Qwen v4 DONE."
touch "$M/.chain_v4_phase1c_qwen_ok"
sleep 30

# ========== Step 3: Phase 1c Llama v4 ==========
log "Phase 1c Llama-3.1-8B v4 (diagnostic, seed=42, per-stratum=100)..."
PYTHONPATH=${TOURMART_ROOT}/src python ${TOURMART_ROOT}/scripts/run_phase1c_crossfamily.py \
    --msgcap-jsonl "$MSGCAP_JSONL" \
    --model-path ${MODELS_DIR_MS}/LLM-Research/Meta-Llama-3___1-8B-Instruct \
    --arm-label llama31_8b_diag_v4 \
    --dtype bfloat16 \
    --diagnostic-window \
    --per-stratum 100 \
    --seed 42 \
    > ${LOGS_DIR}/p1c_v4_llama.log 2>&1 || {
        log "FAIL: Phase 1c Llama v4 exited non-zero"
        touch "$M/.chain_v4_failed_llama"
        exit 1
    }
log "Phase 1c Llama v4 DONE."
touch "$M/.chain_v4_phase1c_llama_ok"

# ========== Final ==========
log "CHAIN v4 COMPLETE. Artifacts:"
ls -lht "$M"/phase1c_*diag_v4* 2>/dev/null | tee -a "$LOG"
touch "$M/.chain_v4_all_done"
log "Wrote .chain_v4_all_done."
