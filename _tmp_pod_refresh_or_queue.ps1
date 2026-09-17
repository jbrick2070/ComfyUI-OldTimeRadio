$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$hostName = "213.173.105.175"
$port = 13317
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$ssh = @("-i", $key, "-p", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25")
$scp = @("-i", $key, "-P", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes")
$text = [System.IO.File]::ReadAllText("$repo\_tmp_pod_refresh_or_queue.sh")
$unix = $text.Replace("`r`n", "`n").Replace("`r", "`n")
[System.IO.File]::WriteAllText("$repo\_tmp_pod_refresh_or_queue.sh.unix", $unix, (New-Object System.Text.UTF8Encoding $false))
scp @scp "$repo\_tmp_pod_refresh_or_queue.sh.unix" "root@${hostName}:/tmp/_otr_refresh_or_queue.sh"
if ($LASTEXITCODE -ne 0) { throw "scp failed rc=$LASTEXITCODE" }
scp @scp "$repo\_tmp_openrouter.secret" "root@${hostName}:/root/.otr_openrouter.env"
if ($LASTEXITCODE -ne 0) { throw "scp openrouter env failed rc=$LASTEXITCODE" }
ssh @ssh "root@${hostName}" "chmod 600 /root/.otr_openrouter.env; bash /tmp/_otr_refresh_or_queue.sh"
if ($LASTEXITCODE -ne 0) { throw "refresh queue failed rc=$LASTEXITCODE" }
