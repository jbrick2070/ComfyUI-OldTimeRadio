# kill_otr_zombies.ps1
# Kills ORPHANED OTR sidecar processes left over from a wedged OTR run.
#
# A process is a target only when BOTH are true:
#   1. It carries a positive OTR marker:
#        * python/pythonw running an OTR worker script (scripts\_otr_*_worker.py,
#          e.g. the Chatterbox and IndexTTS2 sidecars), or
#        * ffmpeg whose command line names an OTR path (otr\episodes, otr\obs,
#          the otr_* temp prefixes, or the pack folder).
#   2. Its parent is provably gone: the parent PID no longer exists, or Windows
#      has reused that PID for a process created AFTER the child.
#
# Never a target: ComfyUI itself (Desktop or headless, any port), Claude / MCP
# helper processes, and anything without an OTR marker. Nothing is selected on
# CPU time, and a missing ComfyUI listener never widens the selection. A process
# whose identity cannot be established (no command line, no creation time, no
# recorded parent) is skipped. Each target's identity is re-checked immediately
# before it is terminated.
#
# Usage (PowerShell):
#   powershell -ExecutionPolicy Bypass -File scripts\kill_otr_zombies.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\kill_otr_zombies.ps1 -Force
# Lists targets first and asks for confirmation; -Force skips the question.
#
# Test mode (kills nothing, ever):
#   powershell -ExecutionPolicy Bypass -File scripts\kill_otr_zombies.ps1 -InventoryPath inv.json
# reads a JSON array of {ProcessId, ParentProcessId, Name, CommandLine,
# CreationDate} records instead of the live process table and prints the
# selection as one "SELECTED_JSON: [...]" line.

param(
    [switch]$Force,
    [string]$InventoryPath = ""
)

$ErrorActionPreference = "Stop"

# Same protection list as otr_reset_gpu.ps1: these are never targets.
$NeverKill = @(
    'Claude Extensions',
    'claude-code',
    'windows-mcp',
    'desktop-commander',
    'ModelContextProtocol'
)

# ComfyUI's own server process (Desktop or headless) runs main.py.
$ComfyMarker = '(^|[\\/"\s])main\.py'

# OTR sidecar worker scripts, e.g. scripts\_otr_chatterbox_worker.py.
$WorkerMarker = '_otr_[A-Za-z0-9_]+_worker\.py'

# Paths only OTR's ffmpeg calls write to or read from.
$FfmpegMarker = '[\\/]otr[\\/](episodes|obs)[\\/]|otr_(ffmpeg|assemble|pcm_probe|cbx|idx2|mesh_tmp)_|ComfyUI-OldTimeRadio'

function ConvertTo-OtrDate {
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [datetime]) { return $Value }
    $text = [string]$Value
    if (-not $text) { return $null }
    try {
        return [datetime]::Parse($text, [Globalization.CultureInfo]::InvariantCulture)
    } catch {
        return $null
    }
}

function Get-LiveInventory {
    $rows = @()
    foreach ($p in (Get-CimInstance Win32_Process)) {
        $rows += [pscustomobject]@{
            ProcessId       = [int]$p.ProcessId
            ParentProcessId = [int]$p.ParentProcessId
            Name            = [string]$p.Name
            CommandLine     = [string]$p.CommandLine
            CreationDate    = ConvertTo-OtrDate $p.CreationDate
        }
    }
    return ,$rows
}

function Get-FileInventory {
    param([string]$Path)
    $rows = @()
    # PowerShell 5.1's ConvertFrom-Json emits a JSON array as ONE object;
    # iterate the parsed value itself, never @(...) around the pipeline.
    $parsed = Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
    foreach ($p in $parsed) {
        $rows += [pscustomobject]@{
            ProcessId       = [int]$p.ProcessId
            ParentProcessId = [int]$p.ParentProcessId
            Name            = [string]$p.Name
            CommandLine     = [string]$p.CommandLine
            CreationDate    = ConvertTo-OtrDate $p.CreationDate
        }
    }
    return ,$rows
}

# 'orphan', 'parented', or 'ambiguous'.
function Get-OrphanState {
    param($Proc, $ByPid)
    if ($null -eq $Proc.CreationDate) { return 'ambiguous' }
    $ppid = [int]$Proc.ParentProcessId
    if ($ppid -le 0) { return 'ambiguous' }
    if (-not $ByPid.ContainsKey($ppid)) { return 'orphan' }
    $parent = $ByPid[$ppid]
    if ($null -eq $parent.CreationDate) { return 'ambiguous' }
    # A "parent" created after the child is a stranger wearing the dead
    # parent's recycled PID, not the process that launched the child.
    if ($parent.CreationDate -gt $Proc.CreationDate) { return 'orphan' }
    return 'parented'
}

function Select-OtrZombies {
    param($Inventory)
    $byPid = @{}
    foreach ($p in $Inventory) { $byPid[[int]$p.ProcessId] = $p }

    $out = @()
    foreach ($p in $Inventory) {
        $name = [string]$p.Name
        $cmd = [string]$p.CommandLine
        if (-not $cmd) { continue }

        $protected = $false
        foreach ($n in $NeverKill) {
            if ($cmd -match [regex]::Escape($n)) { $protected = $true; break }
        }
        if ($protected) { continue }

        $kind = $null
        if ($name -match '^pythonw?\.exe$') {
            if ($cmd -match $ComfyMarker) { continue }
            if ($cmd -match $WorkerMarker) { $kind = 'otr-worker' }
        } elseif ($name -match '^ffmpeg\.exe$') {
            if ($cmd -match $FfmpegMarker) { $kind = 'otr-ffmpeg' }
        }
        if (-not $kind) { continue }

        if ((Get-OrphanState $p $byPid) -ne 'orphan') { continue }

        $out += [pscustomobject]@{
            PID          = [int]$p.ProcessId
            Kind         = $kind
            CreationDate = $p.CreationDate
            FullCommand  = $cmd
            CommandLine  = if ($cmd.Length -gt 120) { $cmd.Substring(0, 120) + "..." } else { $cmd }
        }
    }
    return ,$out
}

# ---------------------------------------------------------------------------
# Test mode: a mocked inventory in, the selection out, nothing terminated.
# ---------------------------------------------------------------------------
if ($InventoryPath) {
    $selected = Select-OtrZombies (Get-FileInventory $InventoryPath)
    $rows = @()
    foreach ($t in $selected) { $rows += [pscustomobject]@{ PID = $t.PID; Kind = $t.Kind } }
    Write-Output ("SELECTED_JSON: " + (ConvertTo-Json -InputObject @($rows) -Compress))
    exit 0
}

$targets = Select-OtrZombies (Get-LiveInventory)

if ($targets.Count -eq 0) {
    Write-Host ""
    Write-Host "No orphaned OTR sidecars found. Nothing to do." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Orphaned OTR sidecars:" -ForegroundColor Cyan
$targets | Select-Object PID, Kind, CommandLine | Format-Table -AutoSize

if (-not $Force) {
    $reply = Read-Host "Kill all $($targets.Count) process(es)? [y/N]"
    if ($reply -ne "y" -and $reply -ne "Y") {
        Write-Host "Aborted. No processes terminated." -ForegroundColor Yellow
        exit 0
    }
}

$killed = 0
foreach ($t in $targets) {
    # Re-check identity right before termination: the PID must still belong
    # to the same process (same command line, same creation time).
    $now = Get-CimInstance Win32_Process -Filter "ProcessId=$($t.PID)" -ErrorAction SilentlyContinue
    if (-not $now) {
        Write-Host "  PID $($t.PID) already exited" -ForegroundColor DarkGray
        continue
    }
    $sameCmd = ([string]$now.CommandLine) -eq $t.FullCommand
    $sameStart = (ConvertTo-OtrDate $now.CreationDate) -eq $t.CreationDate
    if (-not ($sameCmd -and $sameStart)) {
        Write-Host "  PID $($t.PID) changed identity since selection -- skipped" -ForegroundColor Yellow
        continue
    }
    try {
        Stop-Process -Id $t.PID -Force -ErrorAction Stop
        Write-Host "  Killed PID $($t.PID) ($($t.Kind))" -ForegroundColor Green
        $killed++
    } catch {
        Write-Host "  Failed PID $($t.PID): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done. $killed of $($targets.Count) terminated." -ForegroundColor Cyan
