#!/usr/bin/env bash
# Provision one RunPod-style ComfyUI machine through the audited OTR owner.
#
#   ssh root@<ip> -p <port> -i <key> 'bash -s' < scripts/otr_pod_provision.sh
#
# Safe reruns verify exact state. Existing drift is named and left untouched.
set -uo pipefail

CORE_PIN_DEFAULT="169fcf35a2fc163fec31338b816503ddac0d3fcf"
OTR_REPO_URL="https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git"

fail() {
  echo "FATAL: $*" >&2
  exit 1
}

echo "=== OTR pod provision  $(date -u '+%H:%M:%SZ') ==="

# Locate the exact ComfyUI tree, with an explicit override for side-by-side
# installs on templates whose bundled tree is not a git checkout.
COMFY_ROOT="${OTR_COMFY_ROOT:-}"
if [ -z "$COMFY_ROOT" ]; then
  COMFY_ROOT=$(python3 -c "import folder_paths,os;print(os.path.dirname(folder_paths.__file__))" 2>/dev/null || true)
fi
if [ -z "$COMFY_ROOT" ]; then
  FOUND_FOLDER_PATHS=$(find /workspace /app /opt / -maxdepth 5 -name folder_paths.py 2>/dev/null | head -1)
  [ -n "$FOUND_FOLDER_PATHS" ] && COMFY_ROOT=$(dirname "$FOUND_FOLDER_PATHS")
fi
[ -f "$COMFY_ROOT/folder_paths.py" ] || fail "could not locate ComfyUI; set OTR_COMFY_ROOT to its directory"
COMFY_ROOT=$(cd "$COMFY_ROOT" && pwd -P)
export OTR_COMFY_ROOT="$COMFY_ROOT"
CUSTOM_NODES="$COMFY_ROOT/custom_nodes"
mkdir -p "$CUSTOM_NODES"
echo "  comfy root : $COMFY_ROOT"

# Use only an interpreter that proves it imports folder_paths from this tree.
COMFY_PY=""
for candidate in \
  "$COMFY_ROOT/.venv-cu128/bin/python" \
  "$COMFY_ROOT/.venv/bin/python" \
  "$COMFY_ROOT/venv/bin/python" \
  "$COMFY_ROOT"/.venv*/bin/python \
  "$(command -v python3 2>/dev/null || true)"
do
  [ -x "$candidate" ] || continue
  PROBED_ROOT=$(
    cd "$COMFY_ROOT" && "$candidate" -c \
      "import folder_paths,os;print(os.path.realpath(os.path.dirname(folder_paths.__file__)))" \
      2>/dev/null || true
  )
  if [ "$PROBED_ROOT" = "$COMFY_ROOT" ]; then
    COMFY_PY="$candidate"
    break
  fi
done
[ -n "$COMFY_PY" ] || fail "no Python interpreter imports folder_paths from $COMFY_ROOT"
echo "  comfy python: $COMFY_PY"

# The portability lab is defined against one exact ComfyUI core. Never replace
# a template's non-git tree or reset local work. A non-git image gets a clear
# side-by-side recipe instead.
CORE_PIN="${OTR_COMFY_CORE_PIN:-$CORE_PIN_DEFAULT}"
if [ ! -d "$COMFY_ROOT/.git" ]; then
  echo "FATAL: $COMFY_ROOT is not a git checkout; it will not be overwritten." >&2
  echo "Create a pinned side-by-side tree, then rerun against it:" >&2
  echo "  git clone https://github.com/comfyanonymous/ComfyUI.git /workspace/otr-comfyui" >&2
  echo "  export OTR_COMFY_ROOT=/workspace/otr-comfyui" >&2
  echo "  # create/use that tree's Python environment, then rerun this script" >&2
  exit 1
fi
if [ -n "$(git -C "$COMFY_ROOT" status --porcelain --untracked-files=all)" ]; then
  fail "ComfyUI core has tracked changes; refusing to overwrite them"
fi
CURRENT_CORE=$(git -C "$COMFY_ROOT" rev-parse HEAD 2>/dev/null || true)
if [ "$CURRENT_CORE" != "$CORE_PIN" ]; then
  echo "  pin ComfyUI core: $CORE_PIN"
  git -C "$COMFY_ROOT" fetch -q --depth 1 origin "$CORE_PIN" \
    || fail "could not fetch ComfyUI core $CORE_PIN"
  git -C "$COMFY_ROOT" checkout -q --detach FETCH_HEAD \
    || fail "could not check out ComfyUI core $CORE_PIN"
fi
[ "$(git -C "$COMFY_ROOT" rev-parse HEAD 2>/dev/null)" = "$CORE_PIN" ] \
  || fail "ComfyUI core verification failed"
"$COMFY_PY" -m pip install -q -r "$COMFY_ROOT/requirements.txt" \
  || fail "ComfyUI core requirements failed"
echo "  core verified: $CORE_PIN"

# Native Cairo headers are needed by the visualizer's pycairo dependency.
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq >/dev/null 2>&1 \
    && apt-get install -y -qq libcairo2-dev pkg-config >/dev/null 2>&1 \
    || fail "system dependencies libcairo2-dev/pkg-config failed"
fi

# Recover template-scoped model/token settings; SSH does not inherit pid 1's
# environment on common RunPod images. Secrets are never printed.
MODELS_ROOT="${OTR_COMFYUI_MODELS_ROOT:-}"
if [ -z "$MODELS_ROOT" ]; then
  MODELS_ROOT=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null \
    | sed -n 's/^OTR_COMFYUI_MODELS_ROOT=//p' | head -1)
fi
[ -n "$MODELS_ROOT" ] || MODELS_ROOT="$COMFY_ROOT/models"
export OTR_COMFYUI_MODELS_ROOT="$MODELS_ROOT"
export HF_HOME="$MODELS_ROOT/huggingface"
mkdir -p "$HF_HOME"
echo "  models root: $OTR_COMFYUI_MODELS_ROOT"
echo "  HF_HOME    : $HF_HOME"

HF_TOKEN_FILE=/root/.hf_token
TOKEN_VALUE=""
TOKEN_SOURCE=""
if [ -s "$HF_TOKEN_FILE" ]; then
  TOKEN_VALUE=$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")
  TOKEN_SOURCE="$HF_TOKEN_FILE"
elif [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_VALUE=$(printf '%s' "$HF_TOKEN" | tr -d ' \t\r\n')
  TOKEN_SOURCE="this shell"
else
  TOKEN_VALUE=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null \
    | sed -n 's/^HF_TOKEN=//p' | head -1 | tr -d ' \t\r\n')
  [ -n "$TOKEN_VALUE" ] && TOKEN_SOURCE="the pod template"
fi
if [ -n "$TOKEN_VALUE" ]; then
  printf '%s' "$TOKEN_VALUE" > "$HF_TOKEN_FILE"
  chmod 600 "$HF_TOKEN_FILE"
  export HF_TOKEN="$TOKEN_VALUE"
  echo "  HF token   : recovered from $TOKEN_SOURCE (${#TOKEN_VALUE} chars)"
else
  echo "  HF token   : not found; public lanes work, gated manual files need a token"
fi
unset TOKEN_VALUE TOKEN_SOURCE

# Bash owns only the OTR checkout. Python below owns every partner pack,
# dependency, automatic lane, and manual-tier verification.
OTR_ROOT="$CUSTOM_NODES/ComfyUI-OldTimeRadio"
if [ -d "$OTR_ROOT/.git" ]; then
  git -C "$OTR_ROOT" fetch -q origin v2.0-alpha \
    || fail "could not fetch OTR v2.0-alpha"
  git -C "$OTR_ROOT" checkout -q v2.0-alpha \
    || fail "could not select OTR v2.0-alpha (check local changes)"
  git -C "$OTR_ROOT" pull -q --ff-only origin v2.0-alpha \
    || fail "could not fast-forward OTR v2.0-alpha"
elif [ -e "$OTR_ROOT" ]; then
  fail "$OTR_ROOT exists but is not a git checkout; move it aside explicitly"
else
  git clone -q -b v2.0-alpha "$OTR_REPO_URL" "$OTR_ROOT" \
    || fail "could not clone OTR v2.0-alpha"
fi
cd "$OTR_ROOT" || fail "cannot enter OTR checkout"

if ! "$COMFY_PY" scripts/otr_provision.py --packs-only; then
  fail "required node packs or dependencies are incomplete"
fi

# The Python profile router is the sole source of weight ownership. A default
# keeps the entry point useful for strangers; explicit profiles are encouraged
# for HuMo and LTX qualification labs.
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null \
  | head -1 | tr -dc '0-9')
if [ -n "${OTR_PROVISION_PROFILE:-}" ]; then
  PROFILE="$OTR_PROVISION_PROFILE"
elif [ "${VRAM_MIB:-0}" -ge 16000 ] 2>/dev/null; then
  PROFILE="otr_runpod_starter"
else
  PROFILE="otr_nvidia_8gb_haunted"
fi
VOICE_ARGS=()
[ "${OTR_WITH_INDEXTTS2:-0}" = "1" ] && VOICE_ARGS+=(--with-indextts2)
[ "${OTR_WITH_ALL_VOICES:-0}" = "1" ] && VOICE_ARGS=(--with-all-voices)
echo "  profile     : $PROFILE (VRAM ${VRAM_MIB:-unknown} MiB)"
if ! "$COMFY_PY" scripts/otr_provision.py --profile "$PROFILE" "${VOICE_ARGS[@]}"; then
  echo "=== provision INCOMPLETE  $(date -u '+%H:%M:%SZ') ===" >&2
  echo "Read docs/RUNPOD_PORTABILITY_LAB.md for every named manual file." >&2
  exit 1
fi

echo "=== provision complete  $(date -u '+%H:%M:%SZ') ==="
