$ErrorActionPreference = "Stop"
$key = "C:\Users\jeffr\.ssh\runpod_otr"
$hostName = "213.173.105.175"
$port = 13317
$dest = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\_otr_canonical_api_prompt.pod.json"
$scp = @("-i", $key, "-P", "$port", "-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-o", "IdentitiesOnly=yes")
scp @scp "root@${hostName}:/workspace/runpod-slim/ComfyUI/custom_nodes/ComfyUI-OldTimeRadio/scripts/_otr_canonical_api_prompt.json" $dest
if ($LASTEXITCODE -ne 0) { throw "scp prompt dump failed rc=$LASTEXITCODE" }
Write-Output "SAVED $dest"
