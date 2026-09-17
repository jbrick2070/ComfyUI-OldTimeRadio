$ErrorActionPreference = "Stop"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$info = Get-Content -Raw "$repo\_tmp_runpod_ssh.json" | ConvertFrom-Json
$ssh = @(
    "-i", $key, "-p", "$([int]$info.port)",
    "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25",
    "root@$($info.host)"
)
ssh @ssh @'
echo "=== runtime env ==="
if [ -f /workspace/otr-config/otr-runtime.env ]; then
  grep -E '^(COMFY_PY|OTR_|OTR_REPO|OTR_COMFY)=' /workspace/otr-config/otr-runtime.env | sed 's/=.*/=.../' 
  grep -E '^(COMFY_PY|OTR_REPO_ROOT|OTR_COMFY_ROOT)=' /workspace/otr-config/otr-runtime.env
else
  echo NO_RUNTIME_ENV
fi
echo "=== pythons ==="
ls -l /workspace/runpod-slim/ComfyUI/.venv*/bin/python 2>/dev/null || true
ls -l /workspace/runpod-slim/ComfyUI/.venv-py313/bin/python 2>/dev/null || true
find /workspace -maxdepth 5 -type f -name python -path '*/bin/python' 2>/dev/null | head -n 30
echo "=== which ==="
command -v python3 || true
python3 --version || true
echo "=== comfy main ==="
ls -l /workspace/runpod-slim/ComfyUI/main.py /workspace/ComfyUI/main.py 2>/dev/null || true
'@
