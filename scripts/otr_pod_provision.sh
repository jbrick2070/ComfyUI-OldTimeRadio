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
echo "  pip install -r requirements.txt  (into $COMFY_PY)"
"$COMFY_PY" -m pip install -q -r "$CN/ComfyUI-OldTimeRadio/requirements.txt" 2>&1 | tail -2

# 5. Weights for the ungated lane. Needs no Hugging Face token.
echo "  fetching haunted lane weights"
cd "$CN/ComfyUI-OldTimeRadio" || exit 1
"$COMFY_PY" scripts/otr_fetch_lane_weights.py haunted 2>&1 | tail -8

# 5b. THE IMAGE MODEL. Fetching only the video lane is why a machine could
# follow every documented step and still die at OTR_ImageGenDispatcher --
# almost every video lane needs a conditioning still first, and z_image_turbo
# is the image engine for every class in the machine matrix.
#
# Architecture decides WHICH file. nvfp4 needs hardware fp4 (sm_120,
# Blackwell); an Ampere or Ada card cannot execute it, and the adapter ranks
# nvfp4 above bf16 whenever it is present -- so handing the wrong one to an
# older card makes the engine pick the single file that fails.
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.' | tr -d ' ')
if [ "${CC:-0}" -ge 120 ] 2>/dev/null; then ZLANE=z_image_blackwell; else ZLANE=z_image; fi
echo "  fetching image model: $ZLANE (compute_cap ${CC:-unknown})"
"$COMFY_PY" scripts/otr_fetch_lane_weights.py "$ZLANE" 2>&1 | tail -6

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
