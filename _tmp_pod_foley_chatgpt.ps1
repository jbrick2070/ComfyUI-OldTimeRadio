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

$secret = Join-Path $repo "_tmp_openrouter.secret"
if (-not (Test-Path $secret)) { throw "local OpenRouter secret missing" }

To-Unix "$repo\_tmp_pod_foley_chatgpt.sh" "$repo\_tmp_pod_foley_chatgpt.sh.unix"
scp @scp "$repo\_tmp_pod_foley_chatgpt.sh.unix" "root@${hostName}:/tmp/_otr_pod_foley_chatgpt.sh"
if ($LASTEXITCODE -ne 0) { throw "scp launch sh failed rc=$LASTEXITCODE" }
scp @scp $secret "root@${hostName}:/root/.otr_openrouter.env"
if ($LASTEXITCODE -ne 0) { throw "scp openrouter env failed rc=$LASTEXITCODE" }

ssh @ssh "root@${hostName}" "mkdir -p /tmp/_otr_padstrip"
$files = @(
    @{ Local = "nodes\_otr_json.py"; Remote = "/tmp/_otr_padstrip/_otr_json.py" },
    @{ Local = "nodes\_otr_my_story.py"; Remote = "/tmp/_otr_padstrip/_otr_my_story.py" },
    @{ Local = "nodes\otr_shot_lock.py"; Remote = "/tmp/_otr_padstrip/otr_shot_lock.py" },
    @{ Local = "nodes\_otr_slot_drama_contract.py"; Remote = "/tmp/_otr_padstrip/_otr_slot_drama_contract.py" },
    @{ Local = "nodes\_otr_casting.py"; Remote = "/tmp/_otr_padstrip/_otr_casting.py" },
    @{ Local = "nodes\_otr_scifi_news_pro.py"; Remote = "/tmp/_otr_padstrip/_otr_scifi_news_pro.py" },
    @{ Local = "nodes\_otr_video_engines\ghost_signal_author.py"; Remote = "/tmp/_otr_padstrip/ghost_signal_author.py" }
)
foreach ($item in $files) {
    scp @scp (Join-Path $repo $item.Local) ("root@${hostName}:" + $item.Remote)
    if ($LASTEXITCODE -ne 0) { throw ("scp " + $item.Local + " failed rc=" + $LASTEXITCODE) }
}

ssh @ssh "root@${hostName}" "chmod 600 /root/.otr_openrouter.env; bash /tmp/_otr_pod_foley_chatgpt.sh"
if ($LASTEXITCODE -ne 0) { throw "pod foley chatgpt launch failed rc=$LASTEXITCODE" }
