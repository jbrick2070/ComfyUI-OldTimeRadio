$ErrorActionPreference = "Stop"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$info = Get-Content -Raw "$repo\_tmp_runpod_ssh.json" | ConvertFrom-Json
$ssh = @(
    "-i", $key,
    "-p", "$([int]$info.port)",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "ConnectTimeout=25",
    "root@$($info.host)"
)
ssh @ssh "tail -n 80 /workspace/otr-config/ad15_3act_boot.log; echo '---'; nvidia-smi --query-gpu=name,memory.used --format=csv,noheader; curl -fsS http://127.0.0.1:8188/queue || echo QUEUE_DOWN"
