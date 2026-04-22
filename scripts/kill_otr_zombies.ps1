# kill_otr_zombies.ps1
# Kills orphan Python / ffmpeg sidecars left over from a wedged OTR run.
# Preserves the ComfyUI main process (the one owning port 8000).
#
# Usage (PowerShell):
#   cd C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio
#   powershell -ExecutionPolicy Bypass -File scripts\kill_otr_zombies.ps1
#
# Safe by default: lists targets first, asks for confirmation before killing.
# Add -Force to skip the confirmation.

param(
    [switch]$Force
)

$ErrorActionPreference = "Stop"

# 1. Find the ComfyUI main PID (owner of TCP :8000). Never kill this one.
$comfyPid = $null
try {
    $conn = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue |
            Select-Object -First 1
    if ($conn) { $comfyPid = $conn.OwningProcess }
} catch { }

if ($comfyPid) {
    Write-Host "ComfyUI main PID (protected): $comfyPid" -ForegroundColor Green
} else {
    Write-Host "No process listening on :8000 -- ComfyUI may already be down." -ForegroundColor Yellow
    Write-Host "All python.exe processes are candidates for kill." -ForegroundColor Yellow
}

# 2. Enumerate python.exe / pythonw.exe with command lines.
$pyProcs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'"

# 3. Build kill list:
#    - Any python whose command line contains OTR sidecar markers, OR
#    - Any python with > 10s CPU time that is NOT the ComfyUI main PID.
$targets = @()
foreach ($p in $pyProcs) {
    if ($p.ProcessId -eq $comfyPid) { continue }

    $cmd = if ($p.CommandLine) { $p.CommandLine } else { "" }
    $isSidecar = $cmd -match "visual_worker|visual/worker|otr_v2|OTR_VISUAL_BACKEND|io[\\/]visual_in|backends[\\/]"

    # CPU time from the process object
    $procObj = Get-Process -Id $p.ProcessId -ErrorAction SilentlyContinue
    $cpuSec = 0
    if ($procObj) { $cpuSec = [int]$procObj.CPU }

    # Target if: sidecar marker present, OR heavy CPU, OR ComfyUI isn't running (anything python goes)
    $shouldKill = $isSidecar -or ($cpuSec -gt 10) -or (-not $comfyPid)

    if ($shouldKill) {
        $targets += [pscustomobject]@{
            PID         = $p.ProcessId
            CPU_Seconds = $cpuSec
            MemoryMB    = [math]::Round($p.WorkingSetSize / 1MB, 1)
            IsSidecar   = $isSidecar
            CommandLine = if ($cmd.Length -gt 120) { $cmd.Substring(0, 120) + "..." } else { $cmd }
        }
    }
}

# 4. Also grab orphan ffmpeg.exe (no parent or parent is dead)
$ffmpegs = Get-CimInstance Win32_Process -Filter "Name='ffmpeg.exe'"
foreach ($f in $ffmpegs) {
    $parentAlive = $false
    if ($f.ParentProcessId) {
        $parentAlive = [bool](Get-Process -Id $f.ParentProcessId -ErrorAction SilentlyContinue)
    }
    if (-not $parentAlive) {
        $targets += [pscustomobject]@{
            PID         = $f.ProcessId
            CPU_Seconds = 0
            MemoryMB    = [math]::Round($f.WorkingSetSize / 1MB, 1)
            IsSidecar   = $true
            CommandLine = "ffmpeg (orphan): " + $f.CommandLine
        }
    }
}

if ($targets.Count -eq 0) {
    Write-Host ""
    Write-Host "No zombie sidecars found. System is clean." -ForegroundColor Green
    exit 0
}

Write-Host ""
Write-Host "Kill candidates:" -ForegroundColor Cyan
$targets | Format-Table -AutoSize

if (-not $Force) {
    $reply = Read-Host "Kill all $($targets.Count) process(es)? [y/N]"
    if ($reply -ne "y" -and $reply -ne "Y") {
        Write-Host "Aborted. No processes terminated." -ForegroundColor Yellow
        exit 0
    }
}

# 5. Kill them.
$killed = 0
foreach ($t in $targets) {
    try {
        Stop-Process -Id $t.PID -Force -ErrorAction Stop
        Write-Host "  Killed PID $($t.PID)" -ForegroundColor Green
        $killed++
    } catch {
        Write-Host "  Failed PID $($t.PID): $($_.Exception.Message)" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Done. $killed of $($targets.Count) terminated." -ForegroundColor Cyan
if ($comfyPid) {
    $stillAlive = Get-Process -Id $comfyPid -ErrorAction SilentlyContinue
    if ($stillAlive) {
        Write-Host "ComfyUI main (PID $comfyPid) still running." -ForegroundColor Green
    }
}
