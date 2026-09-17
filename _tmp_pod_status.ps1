$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$ssh = @("-o","StrictHostKeyChecking=no","-o","ConnectTimeout=20","-o","BatchMode=yes","-o","IdentitiesOnly=yes","-i",$key,"-p","43125","root@213.173.109.173")
Write-Host "=== queue ==="
& ssh @ssh "curl -fsS http://127.0.0.1:8188/queue"
Write-Host ""
Write-Host "=== obs finals ==="
& ssh @ssh "ls -1 /workspace/runpod-slim/ComfyUI/output/otr/obs/*_final.mp4 2>/dev/null | tail -20"
Write-Host ""
Write-Host "=== git HEAD / pid ==="
& ssh @ssh "cd /workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio; git rev-parse --short HEAD; ps aux | grep -E 'main.py|soak|harness|render' | grep -v grep | head -20"
exit 0
