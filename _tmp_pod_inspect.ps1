$ErrorActionPreference = "Stop"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$ssh = @(
    "-i", $key,
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=20",
    "-p", "43125",
    "root@213.173.109.173"
)
$remote = @'
set -euo pipefail
echo HOST=$(hostname)
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader
echo "=== queue ==="
curl -fsS http://127.0.0.1:8188/queue || echo QUEUE_DOWN
echo
echo "=== git ==="
cd /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio
git rev-parse --abbrev-ref HEAD
git log -1 --oneline
git fetch origin main
echo "--- incoming ---"
git log --oneline HEAD..origin/main | head -n 25
echo "=== obs newest ==="
ls -1t /workspace/runpod-slim/ComfyUI/output/otr/obs/*_final.mp4 2>/dev/null | head -n 8 || true
'@
& ssh @ssh $remote
if ($LASTEXITCODE -ne 0) { throw "pod inspect failed rc=$LASTEXITCODE" }
