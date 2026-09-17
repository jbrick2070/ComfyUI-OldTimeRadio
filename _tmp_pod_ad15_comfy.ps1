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
ssh @ssh "echo '=== boot script ==='; tail -n 20 /workspace/otr-config/ad15_3act_boot.log; echo '=== comfy ==='; tail -n 40 /workspace/otr-config/comfy_8188.log; echo '=== listen ==='; ss -lntp | grep -E '8188|python' || true; echo '=== pid 610 ==='; ps -p 610 -o pid,etime,cmd || true; ps aux | grep -E 'main.py|ad15' | grep -v grep | head"
