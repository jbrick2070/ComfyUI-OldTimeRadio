$ErrorActionPreference = "Stop"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$info = Get-Content -Raw "$repo\_tmp_runpod_ssh.json" | ConvertFrom-Json
$hostName = [string]$info.host
$port = [int]$info.port
$ssh = @("-i", $key, "-p", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25")
$scp = @("-i", $key, "-P", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes")

function Write-Unix([string]$src, [string]$dst) {
    $text = [System.IO.File]::ReadAllText($src)
    $unix = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    [System.IO.File]::WriteAllText($dst, $unix, (New-Object System.Text.UTF8Encoding $false))
}

Write-Unix "$repo\_tmp_pod_ad15_3act.sh" "$repo\_tmp_pod_ad15_3act.sh.unix"
Write-Unix "$repo\_tmp_pod_ad15_submit.py" "$repo\_tmp_pod_ad15_submit.py.unix"

Write-Host "SCP_KEYS"
scp @scp "$repo\_tmp_openrouter.secret" "root@${hostName}:/root/.otr_openrouter.env"
if ($LASTEXITCODE -ne 0) { throw "scp keys failed rc=$LASTEXITCODE" }

Write-Host "SCP_SCRIPTS"
scp @scp "$repo\_tmp_pod_ad15_3act.sh.unix" "root@${hostName}:/tmp/_otr_ad15_3act.sh"
if ($LASTEXITCODE -ne 0) { throw "scp sh failed rc=$LASTEXITCODE" }
scp @scp "$repo\_tmp_pod_ad15_submit.py.unix" "root@${hostName}:/tmp/_otr_ad15_submit.py"
if ($LASTEXITCODE -ne 0) { throw "scp py failed rc=$LASTEXITCODE" }

Write-Host "REMOTE_PULL_BOOT_QUEUE"
ssh @ssh "root@${hostName}" "chmod 600 /root/.otr_openrouter.env; chmod +x /tmp/_otr_ad15_3act.sh; mkdir -p /workspace/otr-config; nohup bash /tmp/_otr_ad15_3act.sh > /workspace/otr-config/ad15_3act_boot.log 2>&1 & echo STARTED=`$!; sleep 3; tail -n 30 /workspace/otr-config/ad15_3act_boot.log"
if ($LASTEXITCODE -ne 0) { throw "ad15 launch failed rc=$LASTEXITCODE" }
