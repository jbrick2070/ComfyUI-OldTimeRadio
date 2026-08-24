<#
.SYNOPSIS
End-to-End Test Suite for 8GB VRAM Profile Matrix on NVIDIA 4060 (Worker)
#>
[CmdletBinding()]
param (
    [int]$ActCount = 1
)

$ErrorActionPreference = "Stop"

$HERE = Split-Path -Parent $MyInvocation.MyCommand.Path
$REPO = Split-Path -Parent $HERE
$LAUNCHCMD = Join-Path $HERE "_otr_soak_server_launch.cmd"
$WATCHDOG = Join-Path $HERE "otr_render_watchdog.ps1"
$APIRUN = Join-Path $HERE "otr_canonical_api_run.py"
$AUDIT = Join-Path $HERE "audit_otr_full_run.py"

$VenvPython = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "venv python not found at $VenvPython"
}

$Legs = @(
    @{
        Label = "Leg 1: LTX 8GB (Core Fast Video + Local LLM)"
        Lane = "LTX"
        Profile = "otr_8gb_ltx"
        SourceBank = "media_archive"
        VisualStyle = "sci_fi_radio"
    },
    @{
        Label = "Leg 2: WAN 8GB (Heavy fp8 ti2v + Local LLM)"
        Lane = "WAN"
        Profile = "otr_8gb_wan"
        SourceBank = "radio_drama"
        VisualStyle = "film_noir"
    },
    @{
        Label = "Leg 3: LTX 8GB + Google API (API LLM Offload)"
        Lane = "LTX"
        Profile = "otr_g4_ltx_8gb"
        SourceBank = "internet_archive"
        VisualStyle = "vintage_comic"
    },
    @{
        Label = "Leg 4: 8GB LITE (Fallback Still Motion/Z-Turbo)"
        Lane = "FLOOR"
        Profile = "8gb_lite"
        SourceBank = "news_broadcast"
        VisualStyle = "pencil_sketch"
    }
)

function Reset-Server {
    Write-Host "[Reset] Ensuring clean slate..." -ForegroundColor Cyan
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
    foreach ($p in $procs) {
        if ($p.CommandLine -match "main\.py.*--port 8000" -or $p.CommandLine -match "otr_soak_server_launch") {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    
    $tcp = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    if ($tcp) {
        Write-Host "Waiting for port 8000 to unbind..." -ForegroundColor Yellow
        Start-Sleep -Seconds 3
    }
    
    # Check baseline VRAM
    $vram = & nvidia-smi --query-gpu=memory.used --format=csv,noheader
    Write-Host "[Reset] Baseline VRAM used: $vram" -ForegroundColor Cyan
}

foreach ($leg in $Legs) {
    Write-Host "`n========================================================" -ForegroundColor Magenta
    Write-Host "Starting $($leg.Label)" -ForegroundColor Magenta
    Write-Host "Profile: $($leg.Profile) | Lane: $($leg.Lane)" -ForegroundColor Magenta
    Write-Host "========================================================" -ForegroundColor Magenta

    Reset-Server
    
    $ServerLog = Join-Path $REPO "otr_server_$($leg.Profile).log"
    $Env:PYTHONUTF8 = "1"
    $Env:PYTHONIOENCODING = "utf-8"
    
    Write-Host "[Boot] Launching ComfyUI server in lane $($leg.Lane)..." -ForegroundColor Cyan
    $pinfo = New-Object System.Diagnostics.ProcessStartInfo
    $pinfo.FileName = $LAUNCHCMD
    $pinfo.Arguments = "`"$ServerLog`" $($leg.Lane)"
    $pinfo.UseShellExecute = $false
    $pinfo.WorkingDirectory = $REPO
    
    $proc = [System.Diagnostics.Process]::Start($pinfo)
    
    Write-Host "[Boot] Waiting for port 8000..." -ForegroundColor Cyan
    $up = $false
    for ($i=0; $i -lt 30; $i++) {
        $tcp = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if ($tcp) {
            $up = $true
            break
        }
        Start-Sleep -Seconds 1
    }
    if (-not $up) {
        Write-Host "Server failed to bind port 8000 in 30s! Check $ServerLog" -ForegroundColor Red
        throw "Boot failure"
    }
    Start-Sleep -Seconds 5 # Let model indexing settle
    
    $ClientLog = Join-Path $REPO "otr_client_$($leg.Profile).log"
    Write-Host "[Run] Triggering API run..." -ForegroundColor Cyan
    
    $args = @(
        $APIRUN,
        "--profile", $leg.Profile,
        "--act-count", $ActCount,
        "--source-bank", $leg.SourceBank,
        "--visual-style", $leg.VisualStyle
    )
    
    $runProc = Start-Process -FilePath $VenvPython -ArgumentList $args -NoNewWindow -PassThru -RedirectStandardOutput $ClientLog -RedirectStandardError $ClientLog
    $runProc.WaitForExit()
    
    if ($runProc.ExitCode -ne 0) {
        Write-Host "API run failed! Exit Code: $($runProc.ExitCode). Check $ClientLog" -ForegroundColor Red
        throw "Run failure"
    }
    
    Write-Host "[Run] API run completed successfully." -ForegroundColor Green
    
    # Find the output episode ID from the client log
    $epId = $null
    $matches = (Select-String -Path $ClientLog -Pattern "\[ledger\] writing manifest to (.*\\episodes\\(.*?)\\).*")
    if ($matches) {
        $epId = $matches.Matches.Groups[2].Value
    }
    if (-not $epId) {
        # Fallback to output parsing
        $matches2 = (Select-String -Path $ClientLog -Pattern "Episode ID:\s*([a-zA-Z0-9_-]+)")
        if ($matches2) {
            $epId = $matches2.Matches.Groups[1].Value
        }
    }
    
    if (-not $epId) {
        Write-Host "Could not find episode ID in $ClientLog for audit." -ForegroundColor Yellow
    } else {
        $EpPath = Join-Path $REPO "output\otr\episodes\$epId"
        if (-not (Test-Path $EpPath)) {
            $EpPath = Join-Path $REPO "otr\episodes\$epId"
        }
        
        Write-Host "[Audit] Auditing episode: $epId at $EpPath" -ForegroundColor Cyan
        $auditProc = Start-Process -FilePath $VenvPython -ArgumentList "$AUDIT --episode `"$EpPath`"" -NoNewWindow -PassThru
        $auditProc.WaitForExit()
        
        if ($auditProc.ExitCode -eq 0) {
            Write-Host "[Audit] PASSED for $($leg.Profile)." -ForegroundColor Green
        } else {
            Write-Host "[Audit] FAILED for $($leg.Profile)." -ForegroundColor Red
            throw "Audit failure"
        }
    }
    
    Write-Host "$($leg.Label) FINISHED SUCCESSFULLY.`n" -ForegroundColor Green
}

Reset-Server
Write-Host "All 8GB profile tests completed successfully." -ForegroundColor Green
