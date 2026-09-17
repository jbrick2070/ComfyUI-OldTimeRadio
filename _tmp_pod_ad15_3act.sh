#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
if [ -f /workspace/otr-config/otr-runtime.env ]; then
  # shellcheck disable=SC1091
  source /workspace/otr-config/otr-runtime.env
fi
if [ -f /root/.otr_openrouter.env ]; then
  chmod 600 /root/.otr_openrouter.env
  sed -i 's/\r$//' /root/.otr_openrouter.env
  set -a
  # shellcheck disable=SC1091
  source /root/.otr_openrouter.env
  set +a
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY//$'\r'/}"
  export OTR_GOOGLE_API_KEY="${OTR_GOOGLE_API_KEY//$'\r'/}"
  export GOOGLE_API_KEY="${GOOGLE_API_KEY//$'\r'/}"
  export GEMINI_API_KEY="${GEMINI_API_KEY//$'\r'/}"
fi
echo OPENROUTER_KEY_CHARS=${#OPENROUTER_API_KEY}
echo GOOGLE_KEY_CHARS=${#OTR_GOOGLE_API_KEY}
if [ "${#OPENROUTER_API_KEY}" -lt 20 ]; then
  echo OPENROUTER_MISSING
  exit 7
fi
if [ "${#OTR_GOOGLE_API_KEY}" -lt 20 ]; then
  echo GOOGLE_MISSING
  exit 7
fi
export OPENROUTER_MODEL_A='~openai/gpt-latest'
export OPENROUTER_MODEL_B='~openai/gpt-mini-latest'
export OPENROUTER_MAX_TOKENS_PER_RUN=800000
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
OTR="${OTR_REPO_ROOT:-/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio}"
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
COMFY_LOG=/workspace/otr-config/comfy_8188.log
BOOT_LOG=/workspace/otr-config/ad15_3act_boot.log
mkdir -p /workspace/otr-config

cd "$OTR"
git fetch origin main
git reset --hard origin/main
git checkout -B main origin/main
echo AFTER=$(git log -1 --oneline)
git rev-parse HEAD
HEAD=$(git rev-parse --short HEAD)

"$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from nodes import _otr_openrouter_backend as orb
print("PY_OR_CHARS", len((os.environ.get("OPENROUTER_API_KEY") or "").strip()))
print("PY_G_CHARS", len((os.environ.get("OTR_GOOGLE_API_KEY") or "").strip()))
if not orb.openrouter_enabled():
    raise SystemExit("OPENROUTER_API_KEY not visible to OTR backend")
catalog = orb.refresh_catalog_cache()
print("source=%s count=%s" % (catalog.get("source"), catalog.get("count")))
if catalog.get("source") != "live":
    raise SystemExit("openrouter catalog refresh was not live")
print("OPENROUTER_LIVE_OK")
PY

pkill -f 'otr_canonical_api_run.py' || true
pkill -f '_tmp_pod_ad15_submit.py' || true
if ss -lntp | grep -q ':8188'; then
  pids=$(ss -lntp | awk '/:8188/{print}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  echo KILL_8188 $pids
  for pid in $pids; do kill "$pid" || true; done
  sleep 4
fi
if ss -lntp | grep -q ':8188'; then
  pids=$(ss -lntp | awk '/:8188/{print}' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u)
  for pid in $pids; do kill -9 "$pid" || true; done
  sleep 2
fi

cd "$OTR_COMFY_ROOT"
: > "$COMFY_LOG"
nohup env \
  OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  OPENROUTER_MODEL_A="$OPENROUTER_MODEL_A" \
  OPENROUTER_MODEL_B="$OPENROUTER_MODEL_B" \
  OPENROUTER_MAX_TOKENS_PER_RUN="$OPENROUTER_MAX_TOKENS_PER_RUN" \
  OTR_GOOGLE_API_KEY="$OTR_GOOGLE_API_KEY" \
  GOOGLE_API_KEY="$GOOGLE_API_KEY" \
  GEMINI_API_KEY="$GEMINI_API_KEY" \
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  "$PY" main.py --listen 0.0.0.0 --port 8188 --disable-metadata \
  > "$COMFY_LOG" 2>&1 &
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
  tail -n 80 "$COMFY_LOG" || true
  exit 4
fi
grep -n "All 25 nodes" "$COMFY_LOG" | tail -n 3 || true
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader || true

cd "$OTR"
env OPENROUTER_API_KEY="$OPENROUTER_API_KEY" \
  OPENROUTER_MODEL_A="$OPENROUTER_MODEL_A" \
  OPENROUTER_MODEL_B="$OPENROUTER_MODEL_B" \
  OPENROUTER_MAX_TOKENS_PER_RUN="$OPENROUTER_MAX_TOKENS_PER_RUN" \
  OTR_GOOGLE_API_KEY="$OTR_GOOGLE_API_KEY" \
  GOOGLE_API_KEY="$GOOGLE_API_KEY" \
  GEMINI_API_KEY="$GEMINI_API_KEY" \
  PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
  COMFYUI_URL=http://127.0.0.1:8188 \
  OTR_REPO_ROOT="$OTR" \
  "$PY" /tmp/_otr_ad15_submit.py
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || true
echo
