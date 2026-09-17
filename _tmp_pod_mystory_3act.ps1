$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$hostName = "213.173.105.175"
$port = 13317
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$ssh = @("-i", $key, "-p", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes", "-o", "ConnectTimeout=25")
$scp = @("-i", $key, "-P", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes")

function To-Unix([string]$src, [string]$dst) {
    $text = [System.IO.File]::ReadAllText($src)
    $unix = $text.Replace("`r`n", "`n").Replace("`r", "`n")
    [System.IO.File]::WriteAllText($dst, $unix, (New-Object System.Text.UTF8Encoding $false))
}

To-Unix "$repo\_tmp_pod_mystory_3act.sh" "$repo\_tmp_pod_mystory_3act.sh.unix"
scp @scp "$repo\_tmp_pod_mystory_3act.sh.unix" "root@${hostName}:/tmp/_otr_pod_mystory_3act.sh"
if ($LASTEXITCODE -ne 0) { throw "scp 3act sh failed rc=$LASTEXITCODE" }
scp @scp "$repo\nodes\_otr_my_story.py" "root@${hostName}:/tmp/_otr_my_story.py"
if ($LASTEXITCODE -ne 0) { throw "scp my_story py failed rc=$LASTEXITCODE" }
ssh @ssh "root@${hostName}" "bash /tmp/_otr_pod_mystory_3act.sh"
if ($LASTEXITCODE -ne 0) { throw "pod 3-act launch failed rc=$LASTEXITCODE" }
