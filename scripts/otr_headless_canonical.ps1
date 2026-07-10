<#
  Safe headless API smoke for the canonical OTR workflow.

  This wrapper exists so agents do not improvise stale harnesses:
    1. Selectively reset only ComfyUI/OTR render Python processes.
    2. Boot the standard ComfyUI API server launcher.
    3. Call scripts\otr_canonical_api_run.py, which always loads
       workflows\otr_canonical.json.

  Examples:
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 `
      -Profile otr_cloud_lanes -Words 30 -Set OTR_LedgerScriptWriter.source_bank=science_news

    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\otr_headless_canonical.ps1 `
      -NoBoot -Profile none -Words 30 -DryRun
#>
param(
    [string]$Profile = "none",
    [int]$Words = 30,
    [string[]]$Set = @(),
    [switch]$DryRun,
    [switch]$NoBoot,
    [switch]$NoReset,
    [int]$Timeout = 5400,
    [int]$PollSeconds = 5,
    [string]$ServerLog = "C:\Users\jeffr\Documents\ComfyUI\comfyui_8000.log"
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$Python = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$Launch = Join-Path $Repo "scripts\_otr_soak_server_launch.cmd"
$Canonical = Join-Path $Repo "workflows\otr_canonical.json"
$StaleExtraEnv = Join-Path $Repo "scripts\_otr_soak_capstone_results\_marathon_extra_env.cmd"

function Say($Message) {
    Write-Host ("[canonical-headless] {0} {1}" -f (Get-Date -Format HH:mm:ss), $Message)
}

function Stop-OtrPython {
    $patterns = @(
        "ComfyUI\\main.py",
        "ComfyUI/main.py",
        "otr_canonical_api_run.py",
        "otr_headless_canonical.ps1"
    )
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
    foreach ($proc in $procs) {
        $cmd = [string]$proc.CommandLine
        if (-not $cmd) { continue }
        $hit = $false
        foreach ($pattern in $patterns) {
            if ($cmd.IndexOf($pattern, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
                $hit = $true
                break
            }
        }
        if ($hit) {
            Say ("stopping pid={0} cmd={1}" -f $proc.ProcessId, $cmd)
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
    $listeners = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
    foreach ($listenerPid in ($listeners.OwningProcess | Select-Object -Unique)) {
        Say ("stopping :8000 listener pid={0}" -f $listenerPid)
        Stop-Process -Id $listenerPid -Force -ErrorAction SilentlyContinue
    }
    for ($i = 0; $i -lt 10; $i++) {
        $remaining = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if (-not $remaining) { return }
        Start-Sleep -Seconds 1
    }
}

function Wait-Comfy {
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        try {
            $stats = Invoke-WebRequest -Uri "http://127.0.0.1:8000/system_stats" -UseBasicParsing -TimeoutSec 5
            $validator = Invoke-WebRequest -Uri "http://127.0.0.1:8000/object_info/OTR_WorkflowValidator" -UseBasicParsing -TimeoutSec 5
            if ($stats.StatusCode -eq 200 -and $validator.StatusCode -eq 200) {
                Say "server healthy"
                return
            }
        } catch {
        }
    }
    throw "ComfyUI API did not become healthy on :8000; see $ServerLog"
}

Set-Location -LiteralPath $Repo

if (-not (Test-Path -LiteralPath $Canonical)) {
    throw "Canonical workflow missing: $Canonical"
}

Say "canonical workflow: $Canonical"

if (-not $NoBoot) {
    if (-not $NoReset) {
        Say "selective reset"
        Stop-OtrPython
        $remaining = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
        if ($remaining) {
            throw ":8000 is still listening after selective reset"
        }
        try {
            $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1
            Say ("vram after reset: {0} MiB" -f $vram)
        } catch {
            Say "nvidia-smi unavailable; continuing"
        }
    }

    if (Test-Path -LiteralPath $StaleExtraEnv) {
        Say "removing stale extra-env hook before canonical boot"
        Remove-Item -LiteralPath $StaleExtraEnv -Force
    }

    Say "booting ComfyUI API"
    $server = Start-Process -FilePath $Launch -ArgumentList "`"$ServerLog`"" -WindowStyle Hidden -PassThru
    Say ("server pid={0} log={1}" -f $server.Id, $ServerLog)
    Wait-Comfy
} else {
    Say "NoBoot set; using existing ComfyUI API server"
    Wait-Comfy
}

$argsList = @(
    "-u",
    "scripts\otr_canonical_api_run.py",
    "--profile", $Profile,
    "--words", "$Words",
    "--timeout", "$Timeout",
    "--poll-s", "$PollSeconds"
)
foreach ($patch in $Set) {
    $argsList += @("--set", $patch)
}
if ($DryRun) {
    $argsList += "--dry-run"
}

Say ("running: {0} {1}" -f $Python, ($argsList -join " "))
& $Python @argsList
exit $LASTEXITCODE
