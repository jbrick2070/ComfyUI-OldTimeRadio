#!/usr/bin/env bash
# THE ACT LADDER ON A RENTED POD -- every graph at 1 act, then 3, then 5, then
# 6 (the writer's structural ceiling; a request past it delivers the closest
# performable episode, it does not fail).
#
# Lives in the repo so it survives the pod. A pod is rented and then stopped;
# on 2026-09-13 this file and its two siblings existed only under /workspace,
# which means one `runpod stop` from being rewritten from memory.
#
# Run it detached ON the pod, from a checkout at the commit you mean to test:
#   nohup bash scripts/otr_pod_ladder.sh <graph> [<graph> ...] #     > /workspace/otr_ladder.log 2>&1 &
#
#   OTR_LADDER_RUNGS  which act rungs to run (default "1 3 5 6"). Pass
#                     "3 5 6" after swapping onto newer code so a rung already
#                     proven is not paid for twice.
#   OTR_COMFY_ROOT    the ComfyUI tree (default /workspace/runpod-slim/ComfyUI)
#
# Each rung is one `otr_shipping_set_legs.sh` run, so every leg gets its own
# log and the SUMMARY.txt lines carry RESULT, minutes and the obs episode name
# -- otr/obs is the success signal, a green log is not.

set -u
COMFY="${OTR_COMFY_ROOT:-/workspace/runpod-slim/ComfyUI}"
OTR=$COMFY/custom_nodes/ComfyUI-OldTimeRadio
PY=""; for c in /opt/otr-venv313/bin/python "$COMFY"/.venv-py313/bin/python "$COMFY"/.venv*/bin/python; do [ -x "$c" ] && PY="$c" && break; done
[ -n "$PY" ] || { echo "no usable venv under $COMFY"; exit 1; }
OBS=$COMFY/output/otr/obs
URL=http://127.0.0.1:8188
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8 HF_HOME=/workspace/huggingface
cd "$OTR" || exit 1
echo "=== ladder start $(date -u '+%H:%M:%SZ') graphs: $* ==="
for acts in ${OTR_LADDER_RUNGS:-1 3 5 6}; do
  # Watch window per leg grows with the act count; a leg past it is cleared.
  export OTR_ACT_COUNT=$acts OTR_LEG_TIMEOUT=$(( (acts + 1) * 7200 ))
  echo "=== rung: $acts act(s)  $(date -u '+%H:%M:%SZ') ==="
  bash scripts/otr_shipping_set_legs.sh "$URL" "$OBS" "$PY" "$@"
  echo "=== rung $acts done  $(date -u '+%H:%M:%SZ') ==="
done
echo "=== ladder done $(date -u '+%H:%M:%SZ') ==="
