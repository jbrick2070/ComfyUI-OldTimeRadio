#!/usr/bin/env bash
# Unattended overnight: bring the pod up, then 1-act every lane, then 3-act the
# ones that passed.
#
# THIS RUNS ON THE POD, NOT THE WORKSTATION. Driving it over SSH from Windows
# ties an eight-hour run to a laptop staying awake and a tunnel staying up; a
# dropped connection would end the night's work and the operator is paying by
# the second either way. On the pod it survives anything happening at the other
# end.
#
# NOTHING HERE BLOCKS ON A HUMAN and no lane can stop the run: a lane that fails
# is recorded and the sweep moves on, because a failure is a result too -- "does
# this work on rented hardware" is exactly the question being asked.
set -uo pipefail

CD=/workspace/runpod-slim/ComfyUI
REPO=$CD/custom_nodes/ComfyUI-OldTimeRadio
PY=$CD/.venv-cu128/bin/python
RESULTS=/root/overnight_results.txt

eval "$(tr '\0' '\n' < /proc/1/environ | grep -E '^OTR_COMFYUI_MODELS_ROOT=' | sed 's/^/export /')"
export HF_HOME="${OTR_COMFYUI_MODELS_ROOT}/huggingface"
[ -f /root/.hf_token ] && export HF_TOKEN="$(cat /root/.hf_token)"
export PYTHONUNBUFFERED=1

echo "=== OVERNIGHT START $(date -u '+%F %H:%M:%SZ') ===" | tee "$RESULTS"

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
  for i in $(seq 1 90); do
    n=$(ready); [ "${n:-0}" -gt 0 ] && { echo "  server ready (OTR_=$n)"; return 0; }
    sleep 10
  done
  return 1
}

# --- server, carrying the token -------------------------------------------
# Identify by LISTENING PORT -- `pgrep -f ComfyUI/main.py` matches this script's
# own command line and kills the wrong process.
PID=$( (ss -lptn 2>/dev/null || netstat -lptn 2>/dev/null) \
       | grep -oE '8188[^0-9].*LISTEN[[:space:]]+([0-9]+)/' \
       | grep -oE '[0-9]+/$' | tr -d '/' | head -1 )
[ -n "$PID" ] && { echo "  killing old comfy $PID" | tee -a "$RESULTS"; kill -9 "$PID"; sleep 4; }
cd "$CD" || exit 1
nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --enable-cors-header \
  > /workspace/comfyui.log 2>&1 &
echo "  comfy relaunched with HF_TOKEN in its environment" | tee -a "$RESULTS"

# --- warm index-tts's runtime repos while the server boots -----------------
(
  export HF_HOME=/workspace/index-tts/checkpoints/hf_cache
  mkdir -p "$HF_HOME"
  /workspace/index-tts/.venv/bin/python - <<'PYEOF'
from huggingface_hub import snapshot_download
for r in ("facebook/w2v-bert-2.0", "amphion/MaskGCT", "funasr/campplus",
          "nvidia/bigvgan_v2_22khz_80band_256x"):
    try:
        snapshot_download(r); print("  OK     %s" % r, flush=True)
    except Exception as e:
        print("  FAILED %-42s %s" % (r, type(e).__name__), flush=True)
PYEOF
) > /root/itwarm.log 2>&1 &

wait_ready || { echo "SERVER NEVER READY -- aborting" | tee -a "$RESULTS"; exit 1; }
tail -5 /root/itwarm.log | sed 's/^/  warm: /' | tee -a "$RESULTS"

# --- the roster: every w45 lane -------------------------------------------
LANES=$(ls -1 "$REPO"/config/profiles/otr_w45_*.json 2>/dev/null \
        | sed 's|.*/otr_w45_||; s|\.json$||' | sort)
echo "  lanes: $(echo $LANES | wc -w)" | tee -a "$RESULTS"

run_leg() {
  local lane="$1" acts="$2"
  local prof="otr_w45_${lane}"
  local log="/root/leg_${lane}_${acts}act.log"
  wait_ready >/dev/null || { printf "  %-24s %-2sact SKIP (server not ready)\n" "$lane" "$acts" | tee -a "$RESULTS"; return 1; }
  local t0=$(date +%s)
  ( cd "$REPO" && COMFYUI_URL=http://127.0.0.1:8188 "$PY" scripts/otr_canonical_api_run.py \
      --profile "$prof" --act-count "$acts" \
      --source-bank original --visual-style sci_fi_radio --timeout 0 ) > "$log" 2>&1
  local rc=$? t1=$(date +%s)
  local res=$(grep -oE "RESULT (SUCCESS|FAIL)" "$log" | tail -1)
  local why=$(grep -oE "is not usable for role[^.]*|MISSING[_A-Z]*|ERROR node [0-9]+ \([A-Za-z_]+\)" "$log" | head -1 | cut -c1-64)
  printf "  %-24s %-2sact %-14s %5ds  %s\n" "$lane" "$acts" "${res:-NO_RESULT}" "$((t1-t0))" "$why" | tee -a "$RESULTS"
  [ "$res" = "RESULT SUCCESS" ]
}

declare -A PASSED
echo "=== PASS 1: one act, every lane  $(date -u '+%H:%M:%SZ') ===" | tee -a "$RESULTS"
for lane in $LANES; do
  if run_leg "$lane" 1; then PASSED[$lane]=1; fi
done

echo "=== PASS 2: three acts, lanes that passed  $(date -u '+%H:%M:%SZ') ===" | tee -a "$RESULTS"
if [ "${#PASSED[@]}" -eq 0 ]; then
  echo "  no lane passed at 1 act -- nothing to escalate" | tee -a "$RESULTS"
else
  for lane in "${!PASSED[@]}"; do run_leg "$lane" 3; done
fi

echo "=== OVERNIGHT DONE $(date -u '+%F %H:%M:%SZ') ===" | tee -a "$RESULTS"
ls -1 "$CD"/output/otr/obs/*.mp4 2>/dev/null | wc -l | sed 's/^/  episodes in pod obs: /' | tee -a "$RESULTS"
