#!/usr/bin/env bash
# Provision a rented ComfyUI pod with OTR, over SSH, with no human input.
#
#   ssh root@<ip> -p <port> -i <key> 'bash -s' < scripts/otr_pod_provision.sh
#
# Idempotent: safe to re-run. Existing clones are pulled, present weights are
# skipped by the fetcher itself.
#
# WHY THE ENV VAR IS PASSED EXPLICITLY. RunPod sets template environment
# variables on the CONTAINER (pid 1), but an SSH session gets a fresh
# environment and does not inherit them. Verified on a live pod: pid 1 has
# OTR_COMFYUI_MODELS_ROOT while `echo $OTR_COMFYUI_MODELS_ROOT` over SSH is
# empty. Without re-reading it here, `_models_root()` falls back to its Windows
# default and several GB of weights land in a literal `C:\ComfyUI-Models`
# directory that ComfyUI never scans -- reporting success the whole time.
set -uo pipefail

echo "=== OTR pod provision  $(date -u '+%H:%M:%SZ') ==="

# 1. Find the tree ComfyUI actually scans. Do NOT assume /workspace/ComfyUI --
#    this image uses /workspace/runpod-slim/ComfyUI, and the obvious guess cost
#    a round trip the first time.
COMFY_ROOT=$(python3 -c "import folder_paths,os;print(os.path.dirname(folder_paths.__file__))" 2>/dev/null)
if [ -z "$COMFY_ROOT" ]; then
  COMFY_ROOT=$(dirname "$(find /workspace / -maxdepth 5 -name folder_paths.py 2>/dev/null | head -1)")
fi
[ -d "$COMFY_ROOT" ] || { echo "FATAL: could not locate ComfyUI"; exit 1; }

# --------------------------------------------------------------------------- #
# 2b. THE HUGGING FACE TOKEN, WITHOUT MAKING ANYONE WRITE A FILE.
#
# Gated weights need it, and the friction was real: the documented way was to
# echo a secret into /root/.hf_token by hand, on every pod, correctly. Easy to
# skip, easy to fumble, and a fumbled one fails later as a 401 on a download
# rather than as anything that says "token".
#
# THE EASY PATH IS A RUNPOD TEMPLATE ENVIRONMENT VARIABLE. Set HF_TOKEN in the
# template UI once and every pod from it has one. The catch -- the same one that
# bites OTR_COMFYUI_MODELS_ROOT -- is that RunPod sets template variables on the
# CONTAINER (pid 1), and an SSH session gets a fresh environment that does not
# inherit them. So pid 1 is read directly.
#
# Order: an existing file, then this shell, then the container, then a previous
# `hf auth login`. The VALUE is never printed -- only where it came from and how
# long it is, which is enough to diagnose a truncated paste and nothing more.
HF_TOKEN_FILE=/root/.hf_token
_tok=""; _src=""
if [ -s "$HF_TOKEN_FILE" ]; then
  _tok=$(tr -d ' \t\r\n' < "$HF_TOKEN_FILE"); _src="existing $HF_TOKEN_FILE"
elif [ -n "${HF_TOKEN:-}" ]; then
  _tok=$(printf '%s' "$HF_TOKEN" | tr -d ' \t\r\n'); _src="HF_TOKEN in this shell"
else
  _tok=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n 's/^HF_TOKEN=//p' | head -1 | tr -d ' \t\r\n')
  if [ -n "$_tok" ]; then _src="the pod's template environment (pid 1)"
  elif [ -s "$HOME/.cache/huggingface/token" ]; then
    _tok=$(tr -d ' \t\r\n' < "$HOME/.cache/huggingface/token"); _src="a previous hf auth login"
  fi
fi
if [ -n "$_tok" ]; then
  printf '%s' "$_tok" > "$HF_TOKEN_FILE"; chmod 600 "$HF_TOKEN_FILE"
  export HF_TOKEN="$_tok"
  echo "  HF token: found in $_src (${#_tok} chars) -> $HF_TOKEN_FILE"
  case "$_tok" in hf_*) : ;; *) echo "    WARNING: does not start with hf_ -- check for a truncated paste" ;; esac
else
  echo "  HF token: NONE FOUND. Ungated lanes still work; gated ones will 401."
  echo "    Set HF_TOKEN in the RunPod template, or: printf '%s' <token> > $HF_TOKEN_FILE"
fi
unset _tok _src

CN="$COMFY_ROOT/custom_nodes"
echo "  comfy root : $COMFY_ROOT"

# 2. Recover the container's env var; fall back to the standard location.
MODELS_ROOT=$(tr '\0' '\n' < /proc/1/environ 2>/dev/null \
              | sed -n 's/^OTR_COMFYUI_MODELS_ROOT=//p' | head -1)
[ -n "$MODELS_ROOT" ] || MODELS_ROOT="$COMFY_ROOT/models"
export OTR_COMFYUI_MODELS_ROOT="$MODELS_ROOT"
echo "  models root: $OTR_COMFYUI_MODELS_ROOT"

# 3. Node packs. -b v2.0-alpha is mandatory: main is thousands of commits
#    behind and still advertises version 1.0.0.
cd "$CN" || exit 1
for spec in \
  "ComfyUI-OldTimeRadio|-b v2.0-alpha https://github.com/jbrick2070/ComfyUI-OldTimeRadio" \
  "ComfyUI-AnimateDiff-Evolved|https://github.com/Kosinkadink/ComfyUI-AnimateDiff-Evolved"
do
  name="${spec%%|*}"; args="${spec#*|}"
  if [ -d "$name/.git" ]; then
    echo "  pull  $name"; git -C "$name" pull --ff-only 2>&1 | tail -1
  else
    echo "  clone $name"; git clone $args "$name" 2>&1 | tail -1
  fi
done

# 4. Dependencies.
# INSTALL INTO THE INTERPRETER COMFYUI ACTUALLY RUNS, not the system python3.
# The image ships ComfyUI in its own venv (.venv-cu128); `python3 -m pip` put
# requirements somewhere ComfyUI never imports from, so `accelerate` was present
# on disk and missing at runtime -- the writer died 18 s into a rented leg with
# "requires `accelerate`" while requirements.txt had listed it all along.
# SYSTEM LIBS FIRST. pycairo is a C extension: without libcairo2-dev and
# pkg-config, `pip install pycairo` fails at metadata generation, and the pack
# then raises ImportError inside the draw function at RENDER time -- every
# viz_mandala frame and every scope overlay. pip alone cannot fix it.
if command -v apt-get >/dev/null 2>&1; then
  echo "  system libs: libcairo2-dev pkg-config"
  apt-get update -qq >/dev/null 2>&1
  apt-get install -y -qq libcairo2-dev pkg-config >/dev/null 2>&1     || echo "    (apt failed -- pycairo may not build; viz_mandala will refuse)"
fi

COMFY_PY="$COMFY_ROOT/.venv-cu128/bin/python"
[ -x "$COMFY_PY" ] || COMFY_PY=$(ls -1 "$COMFY_ROOT"/.venv*/bin/python 2>/dev/null | head -1)
[ -x "$COMFY_PY" ] || COMFY_PY=python3

# --------------------------------------------------------------------------- #
# 4b. COMFYUI ITSELF HAS A MINIMUM VERSION, AND A POD IMAGE IS A SNAPSHOT.
#
# The ltx25 lane resolves LTXVDualCFGGuider and LTXVModalityGuidance out of
# ComfyUI CORE (comfy_extras/nodes_lt.py). They arrived in commit 57ce8e1a,
# "Add support for LTX 2.5 (#15499)", 2026-08-11, first released in v0.32.0.
# No node pack can supply them, and updating ComfyUI-LTXVideo does nothing.
#
# A rented image shipped v0.26.2 -- two months and eight minor versions behind
# -- and the lane failed at render time with WrapperNodeMissing, seventeen
# minutes in, with every weight on disk and every node pack at its latest
# commit. Nothing in the error pointed at a version. That is why this is a step
# and not a note in a document.
#
# UPGRADE ONLY. A tree already at or above the minimum is left untouched, so
# running this can never move a machine that is current.
OTR_COMFY_MIN_TAG="${OTR_COMFY_MIN_TAG:-v0.32.0}"
if [ -d "$COMFY_ROOT/.git" ]; then
  CUR_TAG=$(git -C "$COMFY_ROOT" describe --tags 2>/dev/null | sed 's/-.*//')
  NEWEST=$(printf '%s\n%s\n' "$CUR_TAG" "$OTR_COMFY_MIN_TAG" | sort -V | tail -1)
  if [ -n "$CUR_TAG" ] && [ "$CUR_TAG" = "$NEWEST" ]; then
    echo "  ComfyUI $CUR_TAG >= $OTR_COMFY_MIN_TAG -- left alone"
  else
    echo "  ComfyUI ${CUR_TAG:-unknown} is BELOW $OTR_COMFY_MIN_TAG -- upgrading"
    git -C "$COMFY_ROOT" fetch --tags -q origin 2>/dev/null
    LATEST=$(git -C "$COMFY_ROOT" tag --sort=-v:refname 2>/dev/null | head -1)
    if [ -n "$LATEST" ]; then
      if git -C "$COMFY_ROOT" checkout -q "$LATEST" 2>/dev/null; then
        echo "    now $(git -C "$COMFY_ROOT" describe --tags 2>/dev/null)"
        "$COMFY_PY" -m pip install -q -r "$COMFY_ROOT/requirements.txt" 2>&1 | tail -2
      else
        echo "    checkout FAILED -- left at ${CUR_TAG:-unknown}"
      fi
    else
      echo "    no tags found -- left at ${CUR_TAG:-unknown}"
    fi
  fi
else
  echo "  ComfyUI is not a git checkout -- cannot check its version"
fi

echo "  pip install -r requirements.txt  (into $COMFY_PY)"
"$COMFY_PY" -m pip install -q -r "$CN/ComfyUI-OldTimeRadio/requirements.txt" 2>&1 | tail -2

# 5. Weights for the ungated lane. Needs no Hugging Face token.
cd "$CN/ComfyUI-OldTimeRadio" || exit 1

# WHICH WEIGHTS, AND WHO DECIDES. Not everyone wants every model, and these are
# tens of gigabytes -- so this fetches ONE video lane and ONE image precision,
# which is the minimum that renders an episode, and nothing else. Override with
#
#   OTR_PROVISION_LANES="haunted minimax_h3"   ./otr_pod_provision.sh
#   OTR_PROVISION_LANES=""                     ./otr_pod_provision.sh   # skip
#
# `otr_fetch_lane_weights.py --list` names every lane. Already-present files are
# reported PRESENT and re-downloaded never, so re-running this is cheap.
VIDEO_LANES="${OTR_PROVISION_LANES-haunted}"

# The image model is chosen for you because choosing WRONG is silent: the
# adapter ranks nvfp4 > fp8 > bf16 and nvfp4 needs hardware fp4 (sm_120), so an
# older card handed nvfp4 picks the one file it cannot execute. Precision is a
# size/offload question, NOT a can-it-run question -- z_image_turbo is the
# low-VRAM lane and is proven at 8 GB. Set OTR_PROVISION_IMAGE_LANE to override,
# or empty to skip.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '. ')
VR=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1 | tr -dc '0-9')
if   [ "${CC:-0}" -ge 120 ]   2>/dev/null; then ZDEFAULT=z_image_blackwell
elif [ "${VR:-0}" -ge 20000 ] 2>/dev/null; then ZDEFAULT=z_image
else                                              ZDEFAULT=z_image_int8
fi
IMAGE_LANE="${OTR_PROVISION_IMAGE_LANE-$ZDEFAULT}"

echo "  weights to fetch: video=[${VIDEO_LANES:-none}] image=[${IMAGE_LANE:-none}]"
echo "    (compute_cap ${CC:-?}, vram ${VR:-?} MiB; override with OTR_PROVISION_LANES / OTR_PROVISION_IMAGE_LANE)"
for L in $VIDEO_LANES $IMAGE_LANE; do
  [ -n "$L" ] || continue
  echo "  --- $L"
  "$COMFY_PY" scripts/otr_fetch_lane_weights.py "$L" 2>&1 | tail -6
done

# INDEXTTS2 ON LINUX USES THE ENV VAR, NOT A CODE CHANGE.
#
# eng_indextts2.py resolves `.venv/Scripts/python.exe` -- the Windows layout --
# and a Linux venv keeps its interpreter at `bin/python`. The obvious fix is to
# branch on os.name in the adapter. DO NOT. The engine's qualified voice routes
# are pinned to an adapter/worker FINGERPRINT, so ANY edit to that file
# invalidates them: changing it un-qualified the shipped Lemmy route
# ('lemmy-indextts2-algenib-cockney-v2', fingerprint d47779386ce91209 ->
# c1b64d5c5f6c2f9f) and the cast row silently fell back to an ordinary draw.
# The episode still rendered, which is what makes it easy to miss.
#
# OTR_INDEXTTS2_VENV exists for exactly this, touches no code, and moves no
# fingerprint.
# ISOLATED VENVS SURVIVE A RECREATE ONLY HALFWAY. Their site-packages sit on
# the network volume -- 33 GB of them -- but `.venv/bin/python` is a symlink
# to an interpreter uv installed under /root, which is CONTAINER disk. Every
# pod recreate wipes it and leaves three dangling venvs that report
# "missing" while their packages are perfectly intact. Reinstalling the base
# interpreter repairs all three at once, in about two seconds, and costs
# nothing when it is already there.
export PATH="/root/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
if command -v uv >/dev/null 2>&1; then
  uv python install 3.11 >/dev/null 2>&1 && echo "  uv python 3.11 present (repairs any dangling isolated venv)"
fi

IT_VENV="$(dirname "$COMFY_ROOT")/index-tts/.venv/bin/python"
[ -x "$IT_VENV" ] || IT_VENV="$COMFY_ROOT/index-tts/.venv/bin/python"
if [ -x "$IT_VENV" ]; then
  echo "  OTR_INDEXTTS2_VENV=$IT_VENV"
  export OTR_INDEXTTS2_VENV="$IT_VENV"
  grep -q OTR_INDEXTTS2_VENV /root/.bashrc 2>/dev/null     || echo "export OTR_INDEXTTS2_VENV=\"$IT_VENV\"" >> /root/.bashrc
fi

echo "=== provision done  $(date -u '+%H:%M:%SZ') ==="
ls -1 "$OTR_COMFYUI_MODELS_ROOT/checkpoints" 2>/dev/null | sed 's/^/  ckpt: /'
