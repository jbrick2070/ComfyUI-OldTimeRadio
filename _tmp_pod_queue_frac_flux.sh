#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /workspace/otr-config/otr-runtime.env
OTR="${OTR_REPO_ROOT:-/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio}"
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
LOG=/workspace/otr-config/foley_mystory_3act_chatgpt.log
COMFY_LOG=/workspace/otr-config/comfy_8188.log

# Stop the previous launcher before it POSTs a z_image prompt.
pkill -f '/tmp/_otr_pod_foley_chatgpt.sh' || true
pkill -f 'otr_canonical_api_run.py' || true
sleep 2

ok=0
for i in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo SERVER_HEALTHY i=$i
    ok=1
    break
  fi
  sleep 8
done
if [ "$ok" -ne 1 ]; then
  echo SERVER_NOT_UP
  tail -n 60 "$COMFY_LOG" || true
  exit 4
fi

# Drop any prompt the first launcher already submitted.
curl -fsS -X POST http://127.0.0.1:8188/queue \
  -H 'Content-Type: application/json' \
  -d '{"clear":true}' >/dev/null || true
echo QUEUE_CLEARED
curl -fsS http://127.0.0.1:8188/queue || true
echo

cd "$OTR"
: > "$LOG"
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
nohup "$PY" scripts/otr_canonical_api_run.py \
  --workflow workflows/variants/otr_16gb_foley.json \
  --profile otr_ltx25_foley_flux2klein \
  --act-count 3 \
  --source-bank my_story \
  --visual-style recur_frac \
  --creative-model openrouter:slot-a \
  --technical-model openrouter:slot-a \
  --set 'OTR_LedgerScriptWriter.openrouter_slot_a_model=~openai/gpt-latest' \
  --set 'OTR_LedgerScriptWriter.openrouter_slot_b_model=~openai/gpt-latest' \
  --comfyui-url http://127.0.0.1:8188 \
  --timeout 0 \
  --run-label foley3_chatgpt_frac_flux \
  > "$LOG" 2>&1 &
echo RUNNER_PID=$!
sleep 20
echo "=== runner ==="
tail -n 80 "$LOG" || true
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || true
echo
echo "=== comfy ==="
grep -E 'got prompt|house DEFAULT_IDEA|All 25 nodes|OpenRouter|openrouter|flux|recur_frac|Error|Traceback' "$COMFY_LOG" | tail -n 40 || tail -n 20 "$COMFY_LOG"
