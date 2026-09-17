#!/bin/bash
set -euo pipefail
echo "=== nvidia ==="
nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader || true
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || echo QUEUE_DOWN
echo
echo "=== runner tail ==="
tail -n 40 /workspace/otr-config/foley_mystory_3act.log || true
echo "=== comfy errors ==="
grep -E 'StructuredCallFailed|ValidationError|Traceback|Error|my_story_|obs_publish|Prompt executed|Exception' /workspace/otr-config/comfy_8188.log | tail -n 60 || true
echo "=== last 40 comfy ==="
tail -n 40 /workspace/otr-config/comfy_8188.log || true
