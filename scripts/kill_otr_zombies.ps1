# kill_otr_zombies.ps1
# Kills ORPHANED OTR sidecar processes left over from a wedged OTR run.
#
# A process is a target only when BOTH are true:
#   1. It carries a positive OTR marker:
#        * python/pythonw whose SCRIPT is an OTR worker (scripts\_otr_*_worker.py,
#          e.g. the Chatterbox and IndexTTS2 sidecars) -- the first argument
#          after the interpreter, as OTR launches them, not merely a mention of
#          a worker file somewhere on the command line, or
#        * ffmpeg -- ffmpeg.exe, or a versioned build such as imageio's
#          ffmpeg-win-x86_64-v7.1.exe, which OTR falls back to -- whose command
#          line names an OTR path (otr\episodes, otr\obs, an OTR temp folder or
#          file such as otr_assemble_*, or the pack folder).
#   2. Its parent is provably gone: the parent PID no longer exists, or Windows
#      has reused that PID for a process created AFTER the child. Creation
#      times are compared in UTC, so a DST change cannot reorder them.
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
# selection as one "SELECTED_JSON: [...]" line. Give CreationDate an offset or
# a Z: a naive time is read as THIS machine's local time, and one inside the
# DST fall-back hour is ambiguous (live CIM times carry their own UTC basis).

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

# OTR sidecar workers, e.g. scripts\_otr_chatterbox_worker.py, AS THE SCRIPT
# python runs: OTR launches them as [python, worker, args...], so the worker is
# the first argument after the interpreter. Allowed in between: python's
# no-value flags (-u, -B, -I, ...), -X/-W with their value, and `--`. -c, -m
# and -V never pass, because then python is not running a worker script.
# `python pylint.py ..\_otr_x_worker.py` does not match.
$WorkerMarker = '^\s*(?:"[^"]*"|\S+)\s+(?:(?:-[bBdEiIOPqRsSuvx]+|-[XW]\s*\S+)\s+)*(?:--\s+)?(?:"(?:[^"]*[\\/])?_otr_[A-Za-z0-9_]+_worker\.py"|(?:\S*[\\/])?_otr_[A-Za-z0-9_]+_worker\.py)(?:\s|$)'

# Any ffmpeg* build, the same prefix rule nodes/_otr_shared/proc.py allows:
# ffmpeg.exe, imageio-ffmpeg's ffmpeg-win-x86_64-v7.1.exe (ffmpeg.py falls back
# to it) or a pinned OTR_FFMPEG such as ffmpeg7.exe. ffprobe/ffplay never match.
$FfmpegName = '^ffmpeg[^\\/]*\.exe$'

# Paths only OTR's ffmpeg calls write to or read from. Every marker starts at a
# path-segment boundary, so `not_otr_cbx_report.wav` or a folder merely
# containing the pack's name does not count.
$FfmpegMarker = '[\\/]otr[\\/](episodes|obs)[\\/]|[\\/]otr_(assemble|pcm_probe|cbx|idx2|mesh_tmp)_|[\\/]ComfyUI-OldTimeRadio[\\/]'

# Always UTC: local wall-clock times repeat an hour at the DST fall-back, which
# could make a live parent look younger than its child.
function ConvertTo-OtrDate {
    param($Value)
    if ($null -eq $Value) { return $null }
    if ($Value -is [datetime]) { return $Value.ToUniversalTime() }
    $text = [string]$Value
    if (-not $text) { return $null }
    try {
        return [datetime]::Parse($text, [Globalization.CultureInfo]::InvariantCulture).ToUniversalTime()
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
        } elseif ($name -match $FfmpegName) {
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
if ($PSBoundParameters.ContainsKey('InventoryPath')) {
    # Passing the parameter at all means test mode. An empty value is an
    # error, never a silent fall-through to the live process table.
    if (-not $InventoryPath) {
        # Plain stderr, not Write-Error: under ErrorActionPreference=Stop that
        # throws and the script would exit 1, never reaching this exit 2.
        [Console]::Error.WriteLine("-InventoryPath was given without a file; nothing was selected or killed.")
        exit 2
    }
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
