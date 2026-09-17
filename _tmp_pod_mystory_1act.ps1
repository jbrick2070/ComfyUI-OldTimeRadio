$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$local = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\_tmp_pod_mystory_1act.sh"
$remote = "/tmp/_otr_pod_mystory_1act.sh"
$text = [System.IO.File]::ReadAllText($local)
$unix = $text.Replace("`r`n", "`n").Replace("`r", "`n")
$unixPath = $local + ".unix"
[System.IO.File]::WriteAllText($unixPath, $unix, (New-Object System.Text.UTF8Encoding $false))
scp -i $key -P 43125 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o IdentitiesOnly=yes $unixPath "root@213.173.109.173:$remote"
if ($LASTEXITCODE -ne 0) { throw "scp failed rc=$LASTEXITCODE" }
ssh -i $key -p 43125 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 root@213.173.109.173 "bash $remote"
if ($LASTEXITCODE -ne 0) { throw "pod 1-act launch failed rc=$LASTEXITCODE" }
