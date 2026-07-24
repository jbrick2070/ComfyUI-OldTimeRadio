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
    [int]$Port = 0,
    [string]$ServerLog = "",
    [string]$Workflow = ""
)

$ErrorActionPreference = "Stop"
$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$Python = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$Launch = Join-Path $Repo "scripts\_otr_soak_server_launch.cmd"
$Canonical = Join-Path $Repo "workflows\otr_canonical.json"
$StaleExtraEnv = Join-Path $Repo "scripts\_otr_soak_capstone_results\_marathon_extra_env.cmd"
$ProcessSelectors = Join-Path $Repo "scripts\otr_headless_process.psm1"

Import-Module -Name $ProcessSelectors -Force -ErrorAction Stop

function Say($Message) {
    Write-Host ("[canonical-headless] {0} {1}" -f (Get-Date -Format HH:mm:ss), $Message)
}

function Stop-OtrPython {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
    foreach ($proc in $procs) {
        $cmd = [string]$proc.CommandLine
        if (-not $cmd) { continue }
        if (
            (Test-OtrHeadlessServerCommand -CommandLine $cmd) -or
            (Test-OtrCanonicalRunnerCommand -CommandLine $cmd)
        ) {
            Say ("stopping pid={0} cmd={1}" -f $proc.ProcessId, $cmd)
            Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
        }
    }
    Start-Sleep -Seconds 2
    for ($i = 0; $i -lt 10; $i++) {
        $remaining = @(
            Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue |
                Where-Object { Test-OtrHeadlessServerCommand -CommandLine ([string]$_.CommandLine) }
        )
        if (-not $remaining) { return }
        Start-Sleep -Seconds 1
    }
    $pids = $remaining | ForEach-Object { $_.ProcessId }
    throw "OTR headless process(es) still running after selective reset: $($pids -join ', ')"
}

function Resolve-OtrHeadlessPort([int]$RequestedPort) {
    if ($RequestedPort -gt 0) {
        $listeners = Get-NetTCPConnection -LocalPort $RequestedPort -State Listen -ErrorAction SilentlyContinue
        if ($listeners) {
            $owner = ($listeners.OwningProcess | Select-Object -Unique) -join ', '
            throw "requested port :$RequestedPort is already listening (pid $owner)"
        }
        return $RequestedPort
    }

    $reservation = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
    try {
        $reservation.Start()
        return ([System.Net.IPEndPoint]$reservation.LocalEndpoint).Port
    } finally {
        $reservation.Stop()
    }
}

function Wait-Comfy([int]$ApiPort) {
    for ($i = 0; $i -lt 90; $i++) {
        Start-Sleep -Seconds 2
        try {
            $stats = Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/system_stats" -UseBasicParsing -TimeoutSec 5
            $validator = Invoke-WebRequest -Uri "http://127.0.0.1:$ApiPort/object_info/OTR_WorkflowValidator" -UseBasicParsing -TimeoutSec 5
            if ($stats.StatusCode -eq 200 -and $validator.StatusCode -eq 200) {
                Say "server healthy on :$ApiPort"
                return
            }
        } catch {
        }
    }
    throw "ComfyUI API did not become healthy on :$ApiPort; see $ServerLog"
}

Set-Location -LiteralPath $Repo

if (-not (Test-Path -LiteralPath $Canonical)) {
    throw "Canonical workflow missing: $Canonical"
}

Say "canonical workflow: $Canonical"

# A capability profile has two distinct artifacts: workflow widget overrides
# and launch-time environment.  The latter must reach the ComfyUI server
# BEFORE it boots; applying it only in otr_canonical_api_run.py changes the
# client process, not the already-running server that owns the engine.  Keep
# this boundary explicit so a low-VRAM profile cannot silently run with the
# engine's wider default frame cap.
$ProfileEnv = @{}
if ($Profile -and $Profile -ne "none") {
    $ProfilePath = Join-Path $Repo ("config\profiles\{0}.json" -f $Profile)
    if (-not (Test-Path -LiteralPath $ProfilePath)) {
        throw "Profile not found: $ProfilePath"
    }
    $ProfileObject = Get-Content -LiteralPath $ProfilePath -Raw | ConvertFrom-Json
    foreach ($property in @($ProfileObject.launch.env.PSObject.Properties)) {
        $name = [string]$property.Name
        $value = [string]$property.Value
        if (-not $name -or $null -eq $property.Value) {
            throw "Profile '$Profile' has an invalid launch.env entry"
        }
        $ProfileEnv[$name] = $value
    }
    if ($ProfileEnv.Count -gt 0) {
        if ($NoBoot) {
            foreach ($name in $ProfileEnv.Keys) {
                $current = [string](Get-Item -Path ("Env:{0}" -f $name) -ErrorAction SilentlyContinue).Value
                if ($current -ne $ProfileEnv[$name]) {
                    throw "-NoBoot server cannot satisfy profile '$Profile' launch.env: " +
                        "$name expected '$($ProfileEnv[$name])', current '$current'"
                }
            }
        } else {
            foreach ($name in $ProfileEnv.Keys) {
                Set-Item -Path ("Env:{0}" -f $name) -Value $ProfileEnv[$name]
                Say ("profile launch env: {0}={1}" -f $name, $ProfileEnv[$name])
            }
        }
    }
}

if ($NoBoot -and $Port -le 0) {
    throw "-NoBoot requires an explicit -Port for the existing ComfyUI API server"
}

if (-not $NoBoot -and -not $NoReset) {
    Say "selective OTR headless reset"
    Stop-OtrPython
    try {
        $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits) | Select-Object -First 1
        Say ("vram after reset: {0} MiB" -f $vram)
    } catch {
        Say "nvidia-smi unavailable; continuing"
    }
}

$EffectivePort = if ($NoBoot) { $Port } else { Resolve-OtrHeadlessPort $Port }
if (-not $ServerLog) {
    $ServerLog = Join-Path $Repo ("tmp\otr_headless_{0}.log" -f $EffectivePort)
}
$env:COMFYUI_URL = "http://127.0.0.1:$EffectivePort"
$env:OTR_HEADLESS_PORT = "$EffectivePort"

if (-not $NoBoot) {
    if (Test-Path -LiteralPath $StaleExtraEnv) {
        Say "removing stale extra-env hook before canonical boot"
        Remove-Item -LiteralPath $StaleExtraEnv -Force
    }

    Say "booting ComfyUI API"
    $server = Start-Process -FilePath $Launch -ArgumentList "`"$ServerLog`"" -WindowStyle Hidden -PassThru
    Say ("server pid={0} port={1} log={2}" -f $server.Id, $EffectivePort, $ServerLog)
    Wait-Comfy $EffectivePort
} else {
    Say "NoBoot set; using existing ComfyUI API server on :$EffectivePort"
    Wait-Comfy $EffectivePort
}

$argsList = @(
    "-u",
    "scripts\otr_canonical_api_run.py",
    "--profile", $Profile,
    "--words", "$Words",
    "--timeout", "$Timeout",
    "--poll-s", "$PollSeconds"
)
if ($Workflow -ne "") {
    $argsList += @("--workflow", $Workflow)
}
foreach ($patch in $Set) {
    $argsList += @("--set", $patch)
}
if ($DryRun) {
    $argsList += "--dry-run"
}

Say ("running: {0} {1}" -f $Python, ($argsList -join " "))
& $Python @argsList
$runnerExit = $LASTEXITCODE
if ($runnerExit -ne 0) {
    Say ("RESULT FAIL canonical_runner_exit={0}" -f $runnerExit)
}
exit $runnerExit
