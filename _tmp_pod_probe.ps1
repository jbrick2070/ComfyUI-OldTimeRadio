$ErrorActionPreference = "Continue"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$ssh = @(
    "-o","BatchMode=yes",
    "-o","IdentitiesOnly=yes",
    "-o","StrictHostKeyChecking=accept-new",
    "-o","ConnectTimeout=12",
    "-i",$key,
    "-p","43125",
    "root@213.173.109.173"
)
Write-Host "=== ssh probe last-known ==="
& ssh @ssh "hostname; nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader; echo POD_OK"
Write-Host "rc=$LASTEXITCODE"
