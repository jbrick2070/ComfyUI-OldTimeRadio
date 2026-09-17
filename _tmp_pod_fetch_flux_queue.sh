#!/bin/bash
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
source /workspace/otr-config/otr-runtime.env
if [ -f /root/.otr_openrouter.env ]; then
  chmod 600 /root/.otr_openrouter.env
  set -a
  # shellcheck disable=SC1091
  source /root/.otr_openrouter.env
  set +a
fi
export OPENROUTER_MODEL_A='~openai/gpt-latest'
export OPENROUTER_MODEL_B='~openai/gpt-latest'
export OPENROUTER_MAX_TOKENS_PER_RUN=800000
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8

OTR="${OTR_REPO_ROOT:-/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio}"
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
MODELS="${OTR_COMFY_ROOT:-/workspace/runpod-slim/ComfyUI}/models"
LOG=/workspace/otr-config/foley_mystory_3act_chatgpt.log
COMFY_LOG=/workspace/otr-config/comfy_8188.log

mkdir -p "$MODELS/diffusion_models" "$MODELS/vae" "$MODELS/unet"

fetch() {
  local url="$1" dest="$2" sha="$3"
  if [ -f "$dest" ]; then
    got=$("$PY" - <<PY
import hashlib, pathlib
p = pathlib.Path("$dest")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)
    if [ "$got" = "$sha" ]; then
      echo HAVE "$(basename "$dest")"
      return 0
    fi
    echo BAD_HASH "$(basename "$dest")" got=$got
    rm -f "$dest"
  fi
  echo FETCH "$(basename "$dest")"
  curl -L --fail --retry 5 --retry-delay 2 \
    -A "Mozilla/5.0 (compatible; OTR-provision/1.0)" \
    -o "$dest.part" "$url"
  mv "$dest.part" "$dest"
  got=$("$PY" - <<PY
import hashlib, pathlib
p = pathlib.Path("$dest")
h = hashlib.sha256()
with p.open("rb") as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b""):
        h.update(chunk)
print(h.hexdigest())
PY
)
  if [ "$got" != "$sha" ]; then
    echo HASH_FAIL "$(basename "$dest")" got=$got want=$sha
    exit 8
  fi
  echo OK "$(basename "$dest")"
}

fetch \
  "https://huggingface.co/Latentiq/FLUX.2-klein-4B-GGUF/resolve/4dc94114f28d56e7b63e7bb624a1c1f20353245b/flux-2-klein-4b-Q4_K_M.gguf" \
  "$MODELS/diffusion_models/flux-2-klein-4b-Q4_K_M.gguf" \
  "0b25d143c8469b342bc5af3bce92b783bf6b0636d285f7b2f75e38af63af9a15"

fetch \
  "https://huggingface.co/Comfy-Org/flux2-dev/resolve/ab9055628ea245000e610f2aa2c96f4746093546/split_files/vae/flux2-vae.safetensors" \
  "$MODELS/vae/flux2-vae.safetensors" \
  "d64f3a68e1cc4f9f4e29b6e0da38a0204fe9a49f2d4053f0ec1fa1ca02f9c4b5"

ln -sfn "$MODELS/diffusion_models/flux-2-klein-4b-Q4_K_M.gguf" \
  "$MODELS/unet/flux-2-klein-4b-Q4_K_M.gguf"
echo FLUX_WEIGHTS_READY
ls -lh "$MODELS/diffusion_models/flux-2-klein-4b-Q4_K_M.gguf" "$MODELS/vae/flux2-vae.safetensors"

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
tail -n 80 "$LOG" || true
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || true
echo
