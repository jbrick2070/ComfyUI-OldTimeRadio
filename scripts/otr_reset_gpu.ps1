# otr_reset_gpu.ps1 -- the CLAUDE.md section 4 reset, as a tool.
#
# The soak/quick-smoke harness boots ONE server, runs all legs against it, and does
# NOT tear it down -- it sits RESIDENT holding most of the card. Section 4 says never
# assume a prior run cleaned up, and tear it down SELECTIVELY by CommandLine before
# launching a fresh leg.
#
# This is NOT kill_otr_zombies.ps1. That script deliberately PRESERVES the process
# owning port 8000 (it hunts orphan sidecars around a live server); section 4 needs
# that server gone. Different jobs, both correct.
#
# Usage (PowerShell):
#   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
#   powershell -ExecutionPolicy Bypass -File scripts\otr_reset_gpu.ps1
#
# Exit 0 = port 8000 is free and VRAM is at or below the baseline ceiling.
# Exit 1 = something is still holding the port or the card; DO NOT launch a leg.

param(
    [int]$Port = 8000,
    # Desktop baseline is ~1.5 GB; allow headroom for whatever else is on the card.
    [int]$BaselineCeilingMiB = 3000,
    [int]$SettleSeconds = 6
)

$ErrorActionPreference = "Continue"

# A blanket `Stop-Process -Name python` also kills the Claude MCP extension pythons
# (Desktop Commander / computer-use) and severs our own tools mid-run. Everything
# below matches on CommandLine, and anything wearing one of these markers is never
# a target no matter what else it matches.
$NeverKill = @(
    'Claude Extensions',
    'claude-code',
    'windows-mcp',
    'desktop-commander',
    'ModelContextProtocol'
)

# What section 4 actually wants gone: the ComfyUI server and the soak/sweep harness.
# `.Path` is BLANK for a half-booted server, so this filters on CommandLine only.
$KillMarkers = @(
    'ComfyUI.*main\.py',
    'otr_soak',
    'otr_sweep',
    '_otr_headless',
    'run_video_arm_bakeoff',
    'run_wan_ti2v_bakeoff'
)

function Get-ResetTargets {
    $procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"
    $out = @()
    foreach ($p in $procs) {
        $cmd = if ($p.CommandLine) { $p.CommandLine } else { "" }
        if (-not $cmd) { continue }

        $protected = $false
        foreach ($n in $NeverKill) { if ($cmd -match [regex]::Escape($n)) { $protected = $true; break } }
        if ($protected) { continue }

        $matched = $null
        foreach ($m in $KillMarkers) { if ($cmd -match $m) { $matched = $m; break } }
        if (-not $matched) { continue }

        $out += [pscustomobject]@{
            PID         = $p.ProcessId
            MemoryMB    = [math]::Round($p.WorkingSetSize / 1MB, 1)
            Matched     = $matched
            CommandLine = if ($cmd.Length -gt 110) { $cmd.Substring(0, 110) + "..." } else { $cmd }
        }
    }
    return $out
}

Write-Host "=== OTR GPU reset (CLAUDE.md section 4) ===" -ForegroundColor Cyan

$targets = Get-ResetTargets
if ($targets.Count -eq 0) {
    Write-Host "No ComfyUI server or harness python found." -ForegroundColor Green
} else {
    Write-Host "Tearing down $($targets.Count) process(es):" -ForegroundColor Yellow
    $targets | Format-Table -AutoSize
    foreach ($t in $targets) {
        try {
            Stop-Process -Id $t.PID -Force -ErrorAction Stop
            Write-Host "  killed PID $($t.PID)" -ForegroundColor Green
        } catch {
            Write-Host "  FAILED  PID $($t.PID): $($_.Exception.Message)" -ForegroundColor Red
        }
    }
}

# The port kill: anything still listening after the CommandLine sweep.
$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    $owner = $c.OwningProcess
    $ocmd = (Get-CimInstance Win32_Process -Filter "ProcessId=$owner" -ErrorAction SilentlyContinue).CommandLine
    $protected = $false
    foreach ($n in $NeverKill) { if ($ocmd -and ($ocmd -match [regex]::Escape($n))) { $protected = $true; break } }
    if ($protected) {
        Write-Host "  port $Port held by a PROTECTED process (PID $owner) -- not killed" -ForegroundColor Red
        continue
    }
    try {
        Stop-Process -Id $owner -Force -ErrorAction Stop
        Write-Host "  killed port-$Port owner PID $owner" -ForegroundColor Green
    } catch {
        Write-Host "  FAILED port-$Port owner PID $owner : $($_.Exception.Message)" -ForegroundColor Red
    }
}

Start-Sleep -Seconds $SettleSeconds

# --- Verification. Section 4 requires BOTH of these before a fresh boot. ---
$ok = $true

$still = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($still) {
    Write-Host "PORT $Port STILL LISTENING (PID $($still.OwningProcess))" -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "port $Port : free" -ForegroundColor Green
}

$vram = $null
try {
    $raw = & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>$null
    if ($raw) { $vram = [int](($raw | Select-Object -First 1).Trim()) }
} catch { }

if ($null -eq $vram) {
    Write-Host "VRAM      : could not read nvidia-smi" -ForegroundColor Yellow
} elseif ($vram -gt $BaselineCeilingMiB) {
    Write-Host "VRAM      : $vram MiB -- ABOVE the $BaselineCeilingMiB MiB baseline ceiling" -ForegroundColor Red
    $ok = $false
} else {
    Write-Host "VRAM      : $vram MiB (at baseline)" -ForegroundColor Green
}

if ($ok) {
    Write-Host "RESET OK -- safe to boot a fresh leg." -ForegroundColor Cyan
    exit 0
} else {
    Write-Host "RESET INCOMPLETE -- do not launch a leg." -ForegroundColor Red
    exit 1
}
