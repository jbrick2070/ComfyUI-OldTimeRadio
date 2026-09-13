# Unattended: wait for the in-flight leg, then run the six LTX graphs of the
# shipping set against the server that is already up, one act each.
#
# WHY A SCRIPT AND NOT A SEQUENCE OF TOOL CALLS. Each step is gated on the
# machine's clock, not mine -- a render ending, a server answering. Written
# down, the sequence survives the session that started it and leaves a summary
# the morning can read.
#
# WHY THESE SIX ON THIS BOX. otr_{8gb,16gb}_{video,foley,mime} all render on
# LTX -- 0.9.8 for the 8 GB video tier, 2.5 for the rest -- and every LTX 2.5
# artifact is already on this disk, byte-verified against MANUAL_TIERS. The
# pod gets the ungated six.
#
# WHY THE SERVER IS REUSED, NOT REBOOTED. scripts/_otr_soak_server_launch.cmd
# never passes --use-sage-attention, so a server it booted is sage-free by
# construction; the "SageAttention" line in its log is SeedVR2 reporting the
# LIBRARY is installed, which is not what assert_sage_not_patched tests. The
# sweep submitted 37 legs to this same server without rebooting between them,
# and its LTX 2.5 legs rendered. A warm server also keeps the writer cached.
# If it is NOT answering, boot one the documented way.
param(
    [int]$WaitForPid = 8076,
    [string]$Url = "http://127.0.0.1:8000"
)
$ErrorActionPreference = "Continue"
$root = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location $root
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$night = Join-Path $root "otr\legs\night_5080_$stamp.log"
New-Item -ItemType Directory -Force -Path (Split-Path $night) | Out-Null
function Say($m) { $l = "[{0}] {1}" -f (Get-Date -Format 'HH:mm:ss'), $m; $l | Out-File -FilePath $night -Append -Encoding utf8; Write-Output $l }
function ServerUp { try { $null = Invoke-RestMethod -Uri "$Url/queue" -TimeoutSec 5; return $true } catch { return $false } }

# ---- 1. let the sweep's last leg finish -----------------------------------
Say "waiting for leg pid $WaitForPid to exit (max 120 min)"
$deadline = (Get-Date).AddMinutes(120)
while ((Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 30 }
if (Get-Process -Id $WaitForPid -ErrorAction SilentlyContinue) { Say "leg still running after 120 min -- NOT killing it; giving up the box tonight"; exit 2 }
Say "leg finished"
Say ("latest obs: " + (Get-ChildItem "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs" -Filter *.mp4 | Sort-Object LastWriteTime -Descending | Select-Object -First 1).Name)

# ---- 2. a server, warm if possible ----------------------------------------
if (ServerUp) {
    Say "reusing the running server at $Url"
} else {
    Say "no server answering -- booting one"
    $cmd = Join-Path $root "scripts\_otr_soak_server_launch.cmd"
    $slog = Join-Path $root "otr\legs\night_5080_server_$stamp.log"
    Start-Process -FilePath $cmd -ArgumentList "`"$slog`"" -WindowStyle Hidden
    $up = $false
    for ($i = 0; $i -lt 36; $i++) { Start-Sleep -Seconds 5; if (ServerUp) { $up = $true; break } }
    if (-not $up) { Say "SERVER DID NOT COME UP in 3 min -- read $slog"; Get-Content $slog -Tail 15 | Out-File -FilePath $night -Append -Encoding utf8; exit 3 }
    Say "server up"
}
Say ("VRAM at start: " + (& nvidia-smi --query-gpu=memory.used --format=csv,noheader))

# ---- 3. the six LTX graphs, cheapest first ----------------------------------
# Called in-process so -Graphs binds as an array; a child `powershell -File`
# would receive the list as one comma-joined string.
$legs = @("otr_8gb_video", "otr_16gb_video", "otr_8gb_mime", "otr_16gb_mime", "otr_8gb_foley", "otr_16gb_foley")
& (Join-Path $root "scripts\otr_shipping_set_legs.ps1") -Url $Url -Graphs $legs 2>&1 | ForEach-Object { Say $_ }
Say "night done"
