$ErrorActionPreference = "Continue"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$targets = @(
    @{ Port = 43125; Host = "213.173.109.173" },
    @{ Port = 36743; Host = "213.173.102.153" }
)
foreach ($t in $targets) {
    Write-Host ("=== ssh {0}:{1} ===" -f $t.Host, $t.Port)
    & ssh -o BatchMode=yes -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=8 -i $key -p $t.Port ("root@" + $t.Host) "hostname; nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader; uptime; echo POD_OK"
    Write-Host ("rc=" + $LASTEXITCODE)
}
