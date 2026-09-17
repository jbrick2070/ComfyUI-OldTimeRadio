#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /workspace/otr-config/otr-runtime.env
OTR="${OTR_REPO_ROOT:-/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio}"
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
LOG=/workspace/otr-config/foley_mystory_1act.log

cd "$OTR"
git fetch origin main
git checkout -B main origin/main
echo AFTER=$(git log -1 --oneline)
HEAD=$(git rev-parse --short HEAD)

"$PY" - <<'PY'
import json
from pathlib import Path
p = Path("/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/workflows/variants/otr_16gb_foley.json")
data = json.loads(p.read_text())
writer = next(n for n in data["nodes"] if n["type"] == "OTR_LedgerScriptWriter")
names = [i["widget"]["name"] for i in writer["inputs"] if isinstance(i.get("widget"), dict)]
bank = writer["widgets_values"][names.index("source_bank")]
print("FOLEY_ON_DISK_BANK=%r" % bank)
if bank != "roll (any eligible bank)":
    raise SystemExit("foley json was pinned; refusing to continue")
print("FOLEY_JSON_UNCHANGED_ROLL")
PY

if ss -lntp | grep -q ':8188'; then
  pids=$(ss -lntp | awk '/:8188/{print}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  echo KILL_8188 $pids
  for pid in $pids; do kill "$pid" || true; done
  sleep 4
fi
if ss -lntp | grep -q ':8188'; then
  pids=$(ss -lntp | awk '/:8188/{print}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  echo KILL_8188_FORCE $pids
  for pid in $pids; do kill -9 "$pid" || true; done
  sleep 2
fi
if ss -lntp | grep -q ':8188'; then
  echo PORT_STILL_UP
  ss -lntp | grep ':8188' || true
  exit 6
fi

pkill -f otr_canonical_api_run.py || true

export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
cd "$OTR_COMFY_ROOT"
: > /workspace/otr-config/comfy_8188.log
nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --disable-metadata \
  > /workspace/otr-config/comfy_8188.log 2>&1 &
echo BOOT_PID=$!
ok=0
for i in $(seq 1 45); do
  if curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
    echo SERVER_HEALTHY i=$i HEAD=$HEAD
    ok=1
    break
  fi
  sleep 8
done
if [ "$ok" -ne 1 ]; then
  echo SERVER_NOT_UP
  tail -n 80 /workspace/otr-config/comfy_8188.log || true
  exit 4
fi
grep -n "All 25 nodes" /workspace/otr-config/comfy_8188.log | tail -n 3 || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader || true

cd "$OTR"
: > "$LOG"
nohup "$PY" scripts/otr_canonical_api_run.py \
  --workflow workflows/variants/otr_16gb_foley.json \
  --act-count 1 \
  --source-bank my_story \
  --comfyui-url http://127.0.0.1:8188 \
  --timeout 0 \
  > "$LOG" 2>&1 &
echo RUNNER_PID=$!
sleep 20
echo "=== runner ==="
tail -n 60 "$LOG" || true
echo "=== comfy ==="
grep -E 'got prompt|house DEFAULT_IDEA|All 25 nodes|my_story|Error|Traceback' /workspace/otr-config/comfy_8188.log | tail -n 40 || tail -n 25 /workspace/otr-config/comfy_8188.log
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || true
echo
