#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /workspace/otr-config/otr-runtime.env
if [ -f /root/.otr_openrouter.env ]; then
  chmod 600 /root/.otr_openrouter.env
  sed -i 's/\r$//' /root/.otr_openrouter.env
  set -a
  # shellcheck disable=SC1091
  source /root/.otr_openrouter.env
  set +a
  export OPENROUTER_API_KEY="${OPENROUTER_API_KEY//$'\r'/}"
fi
echo OPENROUTER_KEY_CHARS=${#OPENROUTER_API_KEY}
export OPENROUTER_MODEL_A='~openai/gpt-latest'
export OPENROUTER_MODEL_B='~openai/gpt-latest'
export OPENROUTER_MAX_TOKENS_PER_RUN=800000
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
OTR="${OTR_REPO_ROOT:-/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio}"
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
LOG=/workspace/otr-config/foley_mystory_3act_chatgpt.log
COMFY_LOG=/workspace/otr-config/comfy_8188.log

cd "$OTR"
"$PY" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from nodes import _otr_openrouter_backend as orb
if not orb.openrouter_enabled():
    raise SystemExit("OPENROUTER_API_KEY not visible to refresh")
catalog = orb.refresh_catalog_cache()
meta = orb.catalog_meta()
print(
    "source=%s count=%s fetched_at=%s cache=%s"
    % (catalog.get("source"), catalog.get("count"),
       catalog.get("fetched_at"), orb._catalog_cache_path())
)
if catalog.get("source") != "live":
    raise SystemExit("openrouter catalog refresh was not live")
PY

pkill -f 'otr_canonical_api_run.py' || true
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
nohup "$PY" main.py --listen 0.0.0.0 --port 8188 --disable-metadata \
  > "$COMFY_LOG" 2>&1 &
echo BOOT_PID=$!
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
  tail -n 80 "$COMFY_LOG" || true
  exit 4
fi

"$PY" - <<'PY'
import json, urllib.request
raw = urllib.request.urlopen("http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter", timeout=30).read()
info = json.loads(raw)
slot = info["OTR_LedgerScriptWriter"]["input"]["required"]["openrouter_slot_a_model"][0]
want = "~openai/gpt-latest"
print("SLOT_A_N", len(slot))
print("HAS_GPT_LATEST", want in slot)
if want not in slot:
    print("SLOT_A_HEAD", slot[:16])
    raise SystemExit("chatgpt slug still missing after refresh")
print("OPENROUTER_DROPDOWNS_OK")
PY

cd "$OTR"
: > "$LOG"
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
sleep 18
echo "=== runner ==="
tail -n 50 "$LOG" || true
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || true
echo
