#!/usr/bin/env bash
# SWAP A RUNNING POD ONTO A NEWER COMMIT, BETWEEN LEGS.
#
# Stops the ladder and the harness BY EXACT PID (a pgrep pattern matches the
# shell issuing it -- that mistake cost two remote shells on 2026-09-13),
# interrupts whatever the server is rendering, fast-forwards the checkout,
# restarts ComfyUI so the new module code is actually loaded, and relaunches
# the ladder on the graphs given.
#
#   ssh ... "OTR_LADDER_RUNGS='3 5 6' bash /workspace/pod_swap.sh <graph>..."
#
# RESTARTING COMFYUI IS THE POINT. A git pull alone changes nothing: the server
# imported the old modules at boot and holds them until it dies.
#
# Lives in the repo so it survives the pod being stopped.

set -u
COMFY=/workspace/runpod-slim/ComfyUI
OTR=$COMFY/custom_nodes/ComfyUI-OldTimeRadio
PY=$COMFY/.venv-py313/bin/python
echo "=== swap $(date -u '+%H:%M:%SZ') ==="
# Which act rungs the relaunched ladder runs. Default is the full 1 3 5 6;
# pass "3 5 6" to swap onto new code WITHOUT redoing a rung already proven.
export OTR_LADDER_RUNGS="${OTR_LADDER_RUNGS:-1 3 5 6}"
# Stop the drivers by exact pid (a pattern could match this shell).
for pid in $(pgrep -f "pod_ladder.sh" ) $(pgrep -f "otr_shipping_set_legs.sh") $(pgrep -f "otr_canonical_api_run.py"); do
  [ "$pid" = "$$" ] && continue
  kill "$pid" 2>/dev/null && echo "stopped $pid ($(ps -o comm= -p "$pid" 2>/dev/null))"
done
sleep 2
curl -s -X POST http://127.0.0.1:8188/interrupt >/dev/null 2>&1
q=$(curl -s http://127.0.0.1:8188/queue | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(len(d["queue_running"]), len(d["queue_pending"]))' 2>/dev/null)
echo "queue after interrupt: $q"
cd "$OTR" && git fetch -q origin main && git checkout -q main 2>/dev/null; git reset -q --hard origin/main && echo "OTR now at $(git log -1 --format='%h %s' | cut -c1-90)"
# Restart ComfyUI so the new module code loads.
for pid in $(pgrep -f "main.py --listen 127.0.0.1 --port 8188"); do kill "$pid" 2>/dev/null; done
sleep 5
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HOME="$COMFY/models/huggingface"
cd "$COMFY"
nohup "$PY" main.py --listen 127.0.0.1 --port 8188 --output-directory "$COMFY/output" > /workspace/comfy_night.log 2>&1 &
for i in $(seq 1 90); do sleep 5; curl -fsS http://127.0.0.1:8188/queue >/dev/null 2>&1 && break; done
if curl -fsS http://127.0.0.1:8188/queue >/dev/null 2>&1; then
  echo "ComfyUI back on 8188 after restart; OTR classes: $(curl -s http://127.0.0.1:8188/object_info | "$PY" -c 'import sys,json; d=json.load(sys.stdin); print(len([k for k in d if k.startswith("OTR_")]))')"
else
  echo "ComfyUI did not come back in 7.5 min"; tail -n 15 /workspace/comfy_night.log; exit 3
fi
nohup bash /workspace/pod_ladder.sh "$@" >> /workspace/otr_ladder.log 2>&1 &
sleep 5
echo "ladder relaunched with: $*"
tail -n 3 /workspace/otr_ladder.log
