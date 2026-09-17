$ErrorActionPreference = "Stop"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$key = Join-Path $env:USERPROFILE ".ssh\runpod_otr"
$info = Get-Content -Raw "$repo\_tmp_runpod_ssh.json" | ConvertFrom-Json
$hostName = [string]$info.host
$port = [int]$info.port
$ssh = @("-i", $key, "-p", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25")
$scp = @("-i", $key, "-P", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes")
$text = [System.IO.File]::ReadAllText("$repo\_tmp_pod_fix_py313.sh").Replace("`r`n", "`n").Replace("`r", "`n")
[System.IO.File]::WriteAllText("$repo\_tmp_pod_fix_py313.sh.unix", $text, (New-Object System.Text.UTF8Encoding $false))
scp @scp "$repo\_tmp_pod_fix_py313.sh.unix" "root@${hostName}:/tmp/_otr_fix_py313.sh"
if ($LASTEXITCODE -ne 0) { throw "scp fix failed" }
ssh @ssh "root@${hostName}" "bash /tmp/_otr_fix_py313.sh"
if ($LASTEXITCODE -ne 0) { throw "fix py313 failed rc=$LASTEXITCODE" }
