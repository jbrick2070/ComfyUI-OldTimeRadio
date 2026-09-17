$ErrorActionPreference = "Stop"
$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$Launch = Join-Path $Repo "scripts\_otr_soak_server_launch.cmd"
$ProcessSelectors = Join-Path $Repo "scripts\otr_headless_process.psm1"
$ServerLog = Join-Path $Repo "tmp\otr_foley_mystory_server.log"
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LegDir = Join-Path $Repo ("otr\legs\foley_mystory_{0}" -f $Stamp)
New-Item -ItemType Directory -Force -Path $LegDir | Out-Null
$BootMarker = Join-Path $LegDir "BOOT.txt"

Import-Module -Name $ProcessSelectors -Force -ErrorAction Stop
Set-Location -LiteralPath $Repo

function Note($m) {
    $line = "{0}  {1}" -f (Get-Date -Format "HH:mm:ss"), $m
    Add-Content -LiteralPath $BootMarker -Value $line -Encoding utf8
    Write-Output $line
}

Note "selective reset of headless ComfyUI (reload house_source skip)"
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
foreach ($proc in $procs) {
    $cmd = [string]$proc.CommandLine
    if (-not $cmd) { continue }
    if (Test-OtrHeadlessServerCommand -CommandLine $cmd) {
        Note ("stopping pid={0}" -f $proc.ProcessId)
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 3
for ($i = 0; $i -lt 15; $i++) {
    $listeners = @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
    if (-not $listeners) { break }
    Start-Sleep -Seconds 1
}
$still = @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
if ($still) { throw "port 8000 still listening after reset" }
try {
    $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1
    Note ("vram after reset: {0} MiB" -f $vram)
} catch {
    Note "nvidia-smi unavailable"
}

Note "booting ComfyUI API on :8000"
$env:OTR_HEADLESS_PORT = "8000"
$server = Start-Process -FilePath $Launch -ArgumentList "`"$ServerLog`"" -WindowStyle Hidden -PassThru
Note ("server launcher pid={0} log={1}" -f $server.Id, $ServerLog)

$healthy = $false
for ($i = 0; $i -lt 90; $i++) {
    Start-Sleep -Seconds 2
    try {
        $stats = Invoke-WebRequest -Uri "http://127.0.0.1:8000/system_stats" -UseBasicParsing -TimeoutSec 5
        $validator = Invoke-WebRequest -Uri "http://127.0.0.1:8000/object_info/OTR_WorkflowValidator" -UseBasicParsing -TimeoutSec 5
        if ($stats.StatusCode -eq 200 -and $validator.StatusCode -eq 200) {
            $healthy = $true
            break
        }
    } catch { }
}
if (-not $healthy) { throw "ComfyUI API did not become healthy on :8000; see $ServerLog" }
Note "server healthy"

$LegPs1 = Join-Path $Repo "scripts\otr_shipping_set_legs.ps1"
Note "queueing 1-act otr_16gb_foley source_bank=my_story"
& powershell -NoProfile -ExecutionPolicy Bypass -File $LegPs1 `
    -Url "http://127.0.0.1:8000" `
    -Graphs "otr_16gb_foley" `
    -ActCount "1" `
    -SourceBank "my_story" `
    -TimeoutSec 0 `
    -ObsDir "C:\Users\jeffr\Documents\ComfyUI\output\otr\obs"
$rc = $LASTEXITCODE
Note ("shipping-set exit rc={0}" -f $rc)
exit $rc
