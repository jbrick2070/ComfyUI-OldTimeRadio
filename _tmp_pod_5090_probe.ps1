$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$hostName = "213.173.105.175"
$port = 13317
$ssh = @(
    "-i", $key, "-p", "$port",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=25"
)
ssh @ssh "root@${hostName}" @'
set -e
echo HOST=$(hostname)
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
echo "--- ports ---"
ss -lntp 2>/dev/null | grep -E ':8188|:22 ' || true
echo "--- queue ---"
curl -fsS --max-time 5 http://127.0.0.1:8188/queue || echo NO_COMFY
echo
echo "--- OTR ---"
OTR=/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio
if [ -d "$OTR/.git" ]; then
  cd "$OTR"
  git log -1 --oneline
  git status -sb | head
else
  echo OTR_MISSING
fi
echo "--- py ---"
ls -l /workspace/runpod-slim/ComfyUI/.venv-py313/bin/python 2>/dev/null || echo NO_VENV
echo "--- last foley log ---"
tail -n 15 /workspace/otr-config/foley_mystory_3act.log 2>/dev/null || echo NO_FOLEY_LOG
'@
if ($LASTEXITCODE -ne 0) { throw "pod probe failed rc=$LASTEXITCODE" }
