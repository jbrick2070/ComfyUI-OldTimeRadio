#!/usr/bin/env bash
# Provision one RunPod-style ComfyUI machine through the audited OTR owner.
#
#   ssh root@<ip> -p <port> -i <key> 'bash -s' < scripts/otr_pod_provision.sh
#
# Safe reruns verify exact state. Existing drift is named and left untouched.
set -uo pipefail

CORE_PIN_DEFAULT="169fcf35a2fc163fec31338b816503ddac0d3fcf"
OTR_REPO_URL="https://github.com/jbrick2070/ComfyUI-OldTimeRadio.git"
COMFY_TORCH_CU128="2.10.0+cu128"
COMFY_TORCHVISION_CU128="0.25.0+cu128"
COMFY_TORCHAUDIO_CU128="2.10.0+cu128"
COMFY_TORCH_CU128_INDEX="https://download.pytorch.org/whl/cu128"

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

# Validate the requested OTR recipe before changing ComfyUI core, installing
# packs, downloading weights, or replacing a prior good runtime receipt. The
# OTR checkout is the only input this plan check needs, so fetch it first and
# fail a typo/incompatible Python selection while the pod is still cheap.
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

# The matrix is the ordinary stranger-facing owner. Explicit profiles remain
# an override for named HuMo/LTX qualification labs; they are never the hidden
# default for a new renter.
VRAM_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null \
  | head -1 | tr -dc '0-9')
PROFILE="${OTR_PROVISION_PROFILE:-}"
MACHINE=""
if [ -n "$PROFILE" ]; then
  PROVISION_ARGS=(--profile "$PROFILE")
  SELECTOR="$PROFILE"
else
  MACHINE="${OTR_PROVISION_MACHINE:-}"
  if [ -z "$MACHINE" ]; then
    [ -n "${VRAM_MIB:-}" ] \
      || fail "could not detect NVIDIA VRAM; set an exact OTR_PROVISION_MACHINE or OTR_PROVISION_PROFILE"
    if [ "$VRAM_MIB" -lt 8000 ] 2>/dev/null; then
      fail "detected ${VRAM_MIB} MiB VRAM, below the supported 8 GB machine floor"
    fi
    if [ "${VRAM_MIB:-0}" -ge 16000 ] 2>/dev/null; then
      MACHINE="16gb"
    elif [ "${VRAM_MIB:-0}" -ge 10000 ] 2>/dev/null; then
      MACHINE="12gb"
    else
      MACHINE="8gb"
    fi
  fi
  PROVISION_ARGS=(--machine "$MACHINE")
  SELECTOR="machine:$MACHINE"
fi
VOICE_ARGS=()
[ "${OTR_WITH_INDEXTTS2:-0}" = "1" ] && VOICE_ARGS+=(--with-indextts2)
[ "${OTR_WITH_ALL_VOICES:-0}" = "1" ] && VOICE_ARGS=(--with-all-voices)
echo "  selection   : $SELECTOR (VRAM ${VRAM_MIB:-unknown} MiB)"
if ! "$COMFY_PY" scripts/otr_provision.py \
    "${PROVISION_ARGS[@]}" "${VOICE_ARGS[@]}" --check-plan; then
  fail "requested recipe has no complete provision plan; runtime receipt was not changed"
fi
echo "  plan verified before core, packs, weights, and receipt publication"

probe_comfy_cuda() {
  "$COMFY_PY" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise RuntimeError("torch cannot see the NVIDIA GPU")
device = torch.cuda.get_device_name(0)
sample = torch.ones((64, 64), dtype=torch.float32, device="cuda")
result = sample @ sample
torch.cuda.synchronize()
print(
    "%s | CUDA %s | %s | matmul %.1f"
    % (torch.__version__, torch.version.cuda, device, result[0, 0].item())
)
PY
}

ensure_compatible_comfy_torch() {
  command -v nvidia-smi >/dev/null 2>&1 || return 0
  nvidia-smi -L >/dev/null 2>&1 || fail "nvidia-smi is installed but no GPU is visible"
  if [ "${CUDA_VISIBLE_DEVICES+x}" = x ] && [ -z "${CUDA_VISIBLE_DEVICES}" ]; then
    fail "CUDA_VISIBLE_DEVICES is empty; refusing to rewrite torch for a deliberately hidden GPU"
  fi

  local torch_version driver_version driver_major probe_output probe_rc constraint_pin
  torch_version=$(
    "$COMFY_PY" - <<'PY' 2>/dev/null || true
import torch
print(torch.__version__)
PY
  )
  [ -n "$torch_version" ] || fail "the selected ComfyUI Python cannot import torch"
  driver_version=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader \
    | head -1 | tr -d '[:space:]')
  driver_major=${driver_version%%.*}

  probe_output=$(probe_comfy_cuda 2>&1)
  probe_rc=$?
  if [ "$probe_rc" -ne 0 ]; then
    if [ "$torch_version" = "2.10.0+cu130" ] \
      && [[ "$driver_major" =~ ^[0-9]+$ ]] \
      && [ "$driver_major" -ge 570 ] \
      && [ "$driver_major" -lt 580 ]; then
      echo "  repair torch: template cu130 cannot initialize on NVIDIA driver $driver_version"
      echo "                install the exact 2.10.0 cu128 trio in $COMFY_PY"
      env -u PIP_CONSTRAINT "$COMFY_PY" -m pip install --upgrade --no-cache-dir \
        --index-url "$COMFY_TORCH_CU128_INDEX" \
        "torch==$COMFY_TORCH_CU128" \
        "torchvision==$COMFY_TORCHVISION_CU128" \
        "torchaudio==$COMFY_TORCHAUDIO_CU128" \
        || fail "driver-compatible ComfyUI torch install failed"
      torch_version="$COMFY_TORCH_CU128"
    else
      echo "$probe_output" >&2
      fail "ComfyUI torch $torch_version cannot initialize driver $driver_version; no audited automatic repair matches this tuple"
    fi
  fi

  # Some RunPod CUDA 13 templates export a process-wide pip constraint even
  # when their persistent venv now contains the audited cu128 build. Leaving
  # it active makes the next unpinned pack requirement silently reinstall the
  # incompatible cu130 wheel.
  if [ -n "${PIP_CONSTRAINT:-}" ] && [ -f "$PIP_CONSTRAINT" ]; then
    constraint_pin=$(sed -n 's/[[:space:]]//g; /^torch==/p' "$PIP_CONSTRAINT" \
      | head -1)
    if [ -n "$constraint_pin" ] && [ "$constraint_pin" != "torch==$torch_version" ]; then
      echo "  ignore template pip constraint: $constraint_pin conflicts with torch==$torch_version"
      unset PIP_CONSTRAINT
    fi
  fi

  probe_output=$(probe_comfy_cuda 2>&1) \
    || fail "ComfyUI torch CUDA verification failed after repair: $probe_output"
  echo "  torch verified: $probe_output"
}

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
if [ -n "$(git -C "$COMFY_ROOT" status --porcelain --untracked-files=no)" ]; then
  fail "ComfyUI core has tracked or staged changes; refusing to overwrite them"
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
ensure_compatible_comfy_torch
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
otr_pid1_env() {
  local wanted="$1" entry
  [[ -r /proc/1/environ ]] || return 1
  while IFS= read -r -d '' entry; do
    if [[ "$entry" == "$wanted="* ]]; then
      printf '%s' "${entry#*=}"
      return 0
    fi
  done < /proc/1/environ
  return 1
}

MODELS_ROOT="${OTR_COMFYUI_MODELS_ROOT:-}"
if [ -z "$MODELS_ROOT" ]; then
  MODELS_ROOT=$(otr_pid1_env OTR_COMFYUI_MODELS_ROOT 2>/dev/null || true)
fi
[ -n "$MODELS_ROOT" ] || MODELS_ROOT="$COMFY_ROOT/models"
export OTR_COMFYUI_MODELS_ROOT="$MODELS_ROOT"
export HF_HOME="$MODELS_ROOT/huggingface"
mkdir -p "$HF_HOME"
echo "  models root: $OTR_COMFYUI_MODELS_ROOT"
echo "  HF_HOME    : $HF_HOME"

HF_TOKEN_FILE="${OTR_HF_TOKEN_FILE:-/root/.hf_token}"
TOKEN_VALUE=""
TOKEN_SOURCE=""
TOKEN_XTRACE_WAS_ON=0
[[ $- == *x* ]] && { TOKEN_XTRACE_WAS_ON=1; set +x; }
if [ -n "${HF_TOKEN:-}" ]; then
  TOKEN_VALUE=$(printf '%s' "$HF_TOKEN" | tr -d ' \t\r\n')
  TOKEN_SOURCE="this shell"
elif [ -s "$HF_TOKEN_FILE" ]; then
  TOKEN_VALUE=$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE")
  TOKEN_SOURCE="$HF_TOKEN_FILE"
else
  TOKEN_VALUE=$(otr_pid1_env HF_TOKEN 2>/dev/null \
    | tr -d ' \t\r\n' || true)
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
[[ "$TOKEN_XTRACE_WAS_ON" -eq 1 ]] && set -x
unset TOKEN_XTRACE_WAS_ON

# Bash owns only the OTR checkout. Python below owns every partner pack,
# dependency, automatic lane, and manual-tier verification.
if ! "$COMFY_PY" scripts/otr_provision.py --packs-only; then
  fail "required node packs or dependencies are incomplete"
fi

# Publish one non-secret runtime receipt for every later pod command. The OTR
# repository path is the exact checkout just fetched and executed above; an
# ambient/template value cannot redirect later runtime commands to a second
# tree. Keep the other explicit operator paths, but derive the ordinary RunPod
# layout from the ComfyUI tree that was actually proved above. OTR_INDEXTTS2_VENV is
# deliberately absent: on Linux the runtime adapter resolves the offline
# wrapper through ComfyUI/index-tts, while the online provisioner must use the
# real vendor venv during downloads.
OTR_REPO_ROOT="$OTR_ROOT"
OTR_INDEXTTS2_ROOT="${OTR_INDEXTTS2_ROOT:-$(dirname "$COMFY_ROOT")/index-tts}"
OTR_INDEXTTS2_DIR="${OTR_INDEXTTS2_DIR:-$OTR_INDEXTTS2_ROOT/checkpoints}"
OTR_INDEXTTS2_WORKER="${OTR_INDEXTTS2_WORKER:-$OTR_ROOT/scripts/_otr_indextts2_worker.py}"
OTR_VOICE_REFERENCE_BANK="${OTR_VOICE_REFERENCE_BANK:-/workspace/otr-config/voice_reference_bank.portable.json}"
OTR_PROVISION_PROFILE="$PROFILE"
OTR_PROVISION_MACHINE="$MACHINE"
OTR_PROVISION_SELECTOR="$SELECTOR"
OTR_PROVISION_GENERATION=$(cat /proc/sys/kernel/random/uuid 2>/dev/null \
  || printf '%s-%s' "$(date -u +%Y%m%dT%H%M%SZ)" "$$")
OTR_HEADLESS_PORT="${OTR_HEADLESS_PORT:-8188}"
OTR_RUNTIME_SECRETS_FILE="${OTR_RUNTIME_SECRETS_FILE:-/workspace/otr-config/otr-secrets.env}"
export OTR_REPO_ROOT COMFY_PY OTR_INDEXTTS2_ROOT OTR_INDEXTTS2_DIR
export OTR_INDEXTTS2_WORKER OTR_VOICE_REFERENCE_BANK OTR_PROVISION_PROFILE
export OTR_PROVISION_MACHINE OTR_PROVISION_SELECTOR OTR_PROVISION_GENERATION
export OTR_HEADLESS_PORT OTR_RUNTIME_SECRETS_FILE

OTR_RUNTIME_ENV="${OTR_RUNTIME_ENV:-/workspace/otr-config/otr-runtime.env}"
RUNTIME_DIR=$(dirname "$OTR_RUNTIME_ENV")
mkdir -p "$RUNTIME_DIR" || fail "could not create runtime receipt directory $RUNTIME_DIR"

# Persist only an audited allowlist in a separate protected file. The ordinary
# runtime receipt contains the path, never a credential value. This bridges
# both RunPod template secrets (pid 1) and a one-time no-echo SSH export into
# later sweep shells without making provider credentials profile requirements:
# logged-in Desktop users may instead supply hidden prompt auth.
SECRETS_DIR=$(dirname "$OTR_RUNTIME_SECRETS_FILE")
mkdir -p "$SECRETS_DIR" || fail "could not create runtime secret directory $SECRETS_DIR"
SECRETS_XTRACE_WAS_ON=0
[[ $- == *x* ]] && { SECRETS_XTRACE_WAS_ON=1; set +x; }
SECRET_KEYS=(HF_TOKEN OTR_COMFY_API_KEY OTR_GOOGLE_API_KEY OPENROUTER_API_KEY)
declare -A STORED_SECRETS=()
if [ -s "$OTR_RUNTIME_SECRETS_FILE" ]; then
  for SECRET_KEY in "${SECRET_KEYS[@]}"; do
    STORED_SECRETS["$SECRET_KEY"]=$(
      OTR_SECRET_LOOKUP_KEY="$SECRET_KEY" bash -c '
        set +x
        unset HF_TOKEN OTR_COMFY_API_KEY OTR_GOOGLE_API_KEY OPENROUTER_API_KEY
        source "$1" || exit 1
        printf "%s" "${!OTR_SECRET_LOOKUP_KEY:-}"
      ' otr-secret-read "$OTR_RUNTIME_SECRETS_FILE"
    ) || fail "could not read the existing protected runtime secrets"
  done
fi
SECRETS_TMP=$(mktemp "$SECRETS_DIR/.otr-secrets.env.XXXXXX") \
  || fail "could not create an atomic runtime secret file"
SECRET_COUNT=0
for SECRET_KEY in "${SECRET_KEYS[@]}"; do
  SECRET_VALUE="${!SECRET_KEY:-}"
  if [ -z "$SECRET_VALUE" ]; then
    SECRET_VALUE=$(otr_pid1_env "$SECRET_KEY" 2>/dev/null || true)
  fi
  if [ -z "$SECRET_VALUE" ]; then
    SECRET_VALUE="${STORED_SECRETS[$SECRET_KEY]:-}"
  fi
  if [ -n "$SECRET_VALUE" ]; then
    printf -v "$SECRET_KEY" '%s' "$SECRET_VALUE"
    export "$SECRET_KEY"
    printf 'export %s=%q\n' "$SECRET_KEY" "$SECRET_VALUE" >> "$SECRETS_TMP" \
      || { rm -f "$SECRETS_TMP"; fail "could not write protected runtime secrets"; }
    SECRET_COUNT=$((SECRET_COUNT + 1))
  fi
done
unset SECRET_KEY SECRET_VALUE SECRET_KEYS STORED_SECRETS
chmod 600 "$SECRETS_TMP" \
  || { rm -f "$SECRETS_TMP"; fail "could not protect runtime secrets"; }
mv -f "$SECRETS_TMP" "$OTR_RUNTIME_SECRETS_FILE" \
  || { rm -f "$SECRETS_TMP"; fail "could not publish runtime secrets"; }
echo "  runtime keys: $OTR_RUNTIME_SECRETS_FILE ($SECRET_COUNT credential name(s), mode 0600)"
[[ "$SECRETS_XTRACE_WAS_ON" -eq 1 ]] && set -x
unset SECRETS_XTRACE_WAS_ON

RUNTIME_TMP=$(mktemp "$RUNTIME_DIR/.otr-runtime.env.XXXXXX") \
  || fail "could not create an atomic runtime receipt"
RUNTIME_KEYS=(
  OTR_COMFY_ROOT OTR_REPO_ROOT COMFY_PY OTR_COMFYUI_MODELS_ROOT HF_HOME
  OTR_INDEXTTS2_ROOT OTR_INDEXTTS2_DIR OTR_INDEXTTS2_WORKER
  OTR_VOICE_REFERENCE_BANK OTR_PROVISION_PROFILE OTR_PROVISION_MACHINE
  OTR_PROVISION_SELECTOR OTR_PROVISION_GENERATION OTR_HEADLESS_PORT
  OTR_RUNTIME_SECRETS_FILE
)
for RUNTIME_KEY in "${RUNTIME_KEYS[@]}"; do
  if ! printf 'export %s=%q\n' "$RUNTIME_KEY" "${!RUNTIME_KEY}" >> "$RUNTIME_TMP"; then
    rm -f "$RUNTIME_TMP"
    fail "could not write runtime receipt $OTR_RUNTIME_ENV"
  fi
done
chmod 600 "$RUNTIME_TMP" \
  || { rm -f "$RUNTIME_TMP"; fail "could not protect runtime receipt"; }
mv -f "$RUNTIME_TMP" "$OTR_RUNTIME_ENV" \
  || { rm -f "$RUNTIME_TMP"; fail "could not publish runtime receipt"; }
echo "  runtime env : $OTR_RUNTIME_ENV (non-secret)"

if ! "$COMFY_PY" scripts/otr_provision.py "${PROVISION_ARGS[@]}" "${VOICE_ARGS[@]}"; then
  echo "=== provision INCOMPLETE  $(date -u '+%H:%M:%SZ') ===" >&2
  echo "Read docs/RUNPOD_INSTALL.md for every named manual file." >&2
  exit 1
fi

# Core and partner-pack requirements run after the first repair probe. Prove
# that none of them changed torch back to a build that imports but cannot
# execute on this driver before claiming the machine is complete.
FINAL_CUDA_PROBE=$(probe_comfy_cuda 2>&1) \
  || fail "final ComfyUI CUDA verification failed after all dependencies: $FINAL_CUDA_PROBE"
echo "  final torch verified: $FINAL_CUDA_PROBE"

echo "=== provision complete  $(date -u '+%H:%M:%SZ') ==="
