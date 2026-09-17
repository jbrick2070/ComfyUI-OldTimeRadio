#!/bin/bash
set +e
echo "=== extra_model_paths ==="
ls -l /workspace/runpod-slim/ComfyUI/extra_model_paths.yaml \
  /workspace/otr-config/extra_model_paths.yaml \
  /workspace/runpod-slim/ComfyUI/ComfyUI/extra_model_paths.yaml 2>/dev/null
echo "=== likely model roots ==="
du -sh /workspace/runpod-slim/ComfyUI/models /workspace/models /workspace/ComfyUI/models /workspace/ComfyUI-Models 2>/dev/null
echo "=== flux names ==="
find /workspace/runpod-slim/ComfyUI/models /workspace/models /workspace/ComfyUI-Models /workspace/otr-models \
  -iname '*flux*' \( -name '*.gguf' -o -name '*.safetensors' -o -name '*.sft' \) 2>/dev/null | head -n 80
echo "=== z_image names ==="
find /workspace/runpod-slim/ComfyUI/models /workspace/models \
  -iname '*z_image*' -o -iname '*z-image*' 2>/dev/null | head -n 20
source /workspace/otr-config/otr-runtime.env 2>/dev/null || true
PY="${COMFY_PY:-/workspace/runpod-slim/ComfyUI/.venv-py313/bin/python}"
echo "=== object_info ==="
"$PY" - <<'PY'
import json, urllib.request
url = "http://127.0.0.1:8188/object_info/OTR_VideoDirector"
try:
    raw = urllib.request.urlopen(url, timeout=30).read()
except Exception as exc:
    print("VIDEO_DIRECTOR_FAIL", exc)
    url = "http://127.0.0.1:8188/object_info"
    raw = urllib.request.urlopen(url, timeout=60).read()
    info = json.loads(raw)
    print("NODE_TYPES", [k for k in info if "Image" in k or "Video" in k or "Shot" in k][:40])
    raise SystemExit(0)
info = json.loads(raw)
node = info["OTR_VideoDirector"]["input"]["required"]
for key in ("announcer_image_model", "character_image_model", "music_image_model"):
    opts = node.get(key, [[]])[0]
    flux = [x for x in opts if "flux" in str(x).lower()]
    print(key, "n=", len(opts), "flux=", flux)
PY
