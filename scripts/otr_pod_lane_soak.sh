#!/usr/bin/env bash
# Continuous 1-act soak across every video lane. Runs until stopped.
#
# A SOAK, NOT A PASS. The earlier sweep walked the roster once, so ten lanes
# that failed on a missing model were never retried after it was installed --
# their result recorded a machine state that no longer existed. This loops, so
# a lane fixed at 03:00 is re-proven on the next round without anyone
# remembering to go back for it.
#
# NOTHING STOPS THE RUN. A failing lane is recorded and the loop moves on;
# "which lanes work on rented hardware" is the question, and a failure answers
# it. Rounds are numbered so a lane that flips between rounds is visible as
# flapping rather than averaged away.
set -uo pipefail

CD=/workspace/runpod-slim/ComfyUI
REPO=$CD/custom_nodes/ComfyUI-OldTimeRadio
PY=$CD/.venv-cu128/bin/python
RESULTS=/root/soak_results.txt
ACTS="${SOAK_ACTS:-1}"

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^OTR_COMFYUI_MODELS_ROOT=' | sed 's/^/export /')"
export HF_HOME="${OTR_COMFYUI_MODELS_ROOT}/huggingface"
[ -f /root/.hf_token ] && export HF_TOKEN="$(cat /root/.hf_token)"
export PYTHONUNBUFFERED=1

ready() {
  "$PY" - <<'PYEOF' 2>/dev/null
import json, urllib.request
try:
    d = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/object_info", timeout=90).read().decode())
    q = json.loads(urllib.request.urlopen("http://127.0.0.1:8188/queue", timeout=30).read().decode())
    busy = len(q.get("queue_running", [])) + len(q.get("queue_pending", []))
    print(sum(1 for k in d if k.startswith("OTR_")) if busy == 0 else 0)
except Exception:
    print(0)
PYEOF
}

wait_ready() {
  for _ in $(seq 1 90); do
    [ "$(ready)" -gt 0 ] 2>/dev/null && return 0
    sleep 10
  done
  return 1
}

boot_server() {
  # Identify by LISTENING PORT: a name pattern matches this script's own
  # command line and kills the wrong process.
  local pid
  pid=$( (ss -lptn 2>/dev/null || netstat -lptn 2>/dev/null) \
         | grep -oE '8188[^0-9].*LISTEN[[:space:]]+([0-9]+)/' \
         | grep -oE '[0-9]+/$' | tr -d '/' | head -1 )
  [ -n "$pid" ] && { kill -9 "$pid"; sleep 4; }
  cd "$CD" || exit 1
  nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
    > /workspace/comfyui.log 2>&1 &
}

LANES=$(ls -1 "$REPO"/config/profiles/otr_w45_*.json 2>/dev/null \
        | sed 's|.*/otr_w45_||; s|\.json$||' | sort)

echo "=== SOAK START $(date -u '+%F %H:%M:%SZ')  acts=$ACTS  lanes=$(echo $LANES | wc -w) ===" | tee -a "$RESULTS"
wait_ready || { echo "  server down at start -- booting" | tee -a "$RESULTS"; boot_server; wait_ready; }

round=0
while true; do
  round=$((round + 1))
  echo "--- ROUND $round  $(date -u '+%H:%M:%SZ') ---" | tee -a "$RESULTS"
  for lane in $LANES; do
    if ! wait_ready; then
      echo "  server unresponsive -- rebooting it" | tee -a "$RESULTS"
      boot_server
      wait_ready || { echo "  could not recover; ending soak" | tee -a "$RESULTS"; exit 1; }
    fi
    log="/root/soak_r${round}_${lane}.log"
    t0=$(date +%s)
    ( cd "$REPO" && COMFYUI_URL=http://127.0.0.1:8188 "$PY" scripts/otr_canonical_api_run.py \
        --profile "otr_w45_${lane}" --act-count "$ACTS" \
        --source-bank original --visual-style sci_fi_radio --timeout 0 ) > "$log" 2>&1
    t1=$(date +%s)
    res=$(grep -oE "RESULT (SUCCESS|FAIL)" "$log" | tail -1)
    why=$(grep -oE "ERROR node [0-9]+ \([A-Za-z_]+\)|is not usable for role [a-z_]+: [a-z_]+" "$log" | head -1 | cut -c1-58)
    printf "  r%-2s %-24s %-14s %5ds  %s\n" "$round" "$lane" "${res:-NO_RESULT}" "$((t1-t0))" "$why" | tee -a "$RESULTS"
    # Keep only the last few logs per lane so a long soak cannot fill the disk.
    ls -1t /root/soak_r*_"${lane}".log 2>/dev/null | tail -n +4 | xargs -r rm -f
  done
  echo "  round $round done -- pod obs: $(ls -1 "$CD"/output/otr/obs/*.mp4 2>/dev/null | wc -l)" | tee -a "$RESULTS"
done
