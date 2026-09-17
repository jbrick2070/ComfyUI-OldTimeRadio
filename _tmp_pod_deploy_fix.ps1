$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$local = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\_tmp_pod_deploy_fix.sh"
$remote = "/tmp/_otr_pod_restart_foley_mystory.sh"
$text = [System.IO.File]::ReadAllText($local)
$unix = $text.Replace("`r`n", "`n").Replace("`r", "`n")
$unixPath = $local + ".unix"
[System.IO.File]::WriteAllText($unixPath, $unix, (New-Object System.Text.UTF8Encoding $false))
scp -i $key -P 43125 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o IdentitiesOnly=yes $unixPath "root@213.173.109.173:$remote"
ssh -i $key -p 43125 -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o IdentitiesOnly=yes -o ConnectTimeout=25 root@213.173.109.173 "bash $remote"
