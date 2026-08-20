# Reset the GPU box per CLAUDE.md section 4, before any headless run.
#
# SELECTIVE BY COMMANDLINE, NEVER A BLANKET PYTHON KILL. A blanket
# `Stop-Process -Name python` also kills the Claude MCP extension pythons
# (Desktop Commander / computer-use) and severs the driving session's own tools
# mid-run. `.Path` is BLANK for a half-booted server, so the filter must match
# on CommandLine.
#
# Exits 0 when the box is verified clean: nothing listening on :8000 and GPU
# memory back to the desktop baseline. Exits 1 if it could not get there, so a
# caller can refuse to launch onto a dirty box rather than discover it later.

$ErrorActionPreference = 'Continue'
$BaselineCeilingMb = 2500      # desktop baseline is ~1.5 GB; allow headroom
$Port = 8000

Write-Output "=== BEFORE ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader

$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        $_.CommandLine -and (
            $_.CommandLine -match 'ComfyUI\\ComfyUI\\main\.py' -or
            $_.CommandLine -match 'otr_canonical_api_run' -or
            $_.CommandLine -match 'otr_soak' -or
            $_.CommandLine -match 'otr_banksweep'
        )
    }

foreach ($p in $targets) {
    Write-Output ("killing pid={0} :: {1}" -f $p.ProcessId,
        $p.CommandLine.Substring(0, [Math]::Min(110, $p.CommandLine.Length)))
    try { Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop } catch {
        Write-Output ("  could not stop {0}: {1}" -f $p.ProcessId, $_)
    }
}

# The port holder may be a process the CommandLine filter missed.
$conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
foreach ($c in $conns) {
    Write-Output ("killing port holder pid={0}" -f $c.OwningProcess)
    try { Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop } catch {}
}

# VRAM is released asynchronously; poll rather than assume.
$clean = $false
for ($i = 0; $i -lt 20; $i++) {
    Start-Sleep -Seconds 2
    $listening = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
    $usedRaw = (nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    $used = 0
    if ($usedRaw) { $used = [int](($usedRaw -split "`n")[0].Trim()) }
    if (-not $listening -and $used -lt $BaselineCeilingMb) { $clean = $true; break }
}

Write-Output "=== AFTER ==="
nvidia-smi --query-gpu=memory.used --format=csv,noheader
$still = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($still) { Write-Output "PORT $Port STILL LISTENING" } else { Write-Output "port $Port clear" }

if ($clean) { Write-Output "RESET OK"; exit 0 }
Write-Output "RESET INCOMPLETE -- do not launch onto this box"
exit 1
