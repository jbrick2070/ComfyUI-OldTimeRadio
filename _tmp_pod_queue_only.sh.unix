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
export OPENROUTER_MODEL_A='~openai/gpt-latest'
export OPENROUTER_MODEL_B='~openai/gpt-latest'
export OPENROUTER_MAX_TOKENS_PER_RUN=800000
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
OTR="${OTR_REPO_ROOT:-/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio}"
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
LOG=/workspace/otr-config/foley_mystory_3act_chatgpt.log

if ! curl -fsS http://127.0.0.1:8188/system_stats >/dev/null 2>&1; then
  echo COMFY_DOWN
  exit 4
fi
echo COMFY_UP
echo OPENROUTER_KEY_CHARS=${#OPENROUTER_API_KEY}

"$PY" - <<'PY'
import json, urllib.request
raw = urllib.request.urlopen(
    "http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter", timeout=30
).read()
info = json.loads(raw)["OTR_LedgerScriptWriter"]["input"]
block = info.get("optional") or info.get("required") or {}
if "openrouter_slot_a_model" not in block:
    print("INPUT_KEYS", sorted((info.get("required") or {}).keys())[:20],
          sorted((info.get("optional") or {}).keys())[:40])
    raise SystemExit("openrouter_slot_a_model missing from object_info")
slot = block["openrouter_slot_a_model"][0]
want = "~openai/gpt-latest"
print("SLOT_A_N", len(slot))
print("HAS_GPT_LATEST", want in slot)
print("SLOT_A_HEAD", slot[:12])
if want not in slot:
    raise SystemExit("chatgpt slug still missing after refresh")
print("OPENROUTER_DROPDOWNS_OK")
PY

pkill -f 'otr_canonical_api_run.py' || true
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
sleep 20
echo "=== runner ==="
tail -n 60 "$LOG" || true
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || true
echo
