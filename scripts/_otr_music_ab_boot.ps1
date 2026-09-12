# Reset and boot ONE headless server for a music A/B arm.
#
# Called by scripts/otr_music_ab.py, which puts the arm's OTR_SA3_* settings in
# this process's environment; Start-Process passes them to the launcher and the
# launcher to ComfyUI, which is the only place those settings are read.
#
# WHY A .ps1 AND NOT subprocess.Popen (measured 2026-09-12, both ways failed):
#   * `wmic` is gone on Windows 11, so a wmic-based kill silently does nothing,
#     the old server keeps :8000, and the new one cannot bind.
#   * Popen on a .cmd with DETACHED_PROCESS returns a pid and never runs the
#     batch -- no log, no server, no error.
# CLAUDE.md section 5 says launch the .cmd via Start-Process -FilePath and never
# through a cmd.exe /c whose two-quoted-token rule eats the log path. This is
# that, plus the section 4 selective reset.
param(
    [Parameter(Mandatory = $true)][string]$Launcher,
    [Parameter(Mandatory = $true)][string]$LogPath,
    [int]$TimeoutSeconds = 300
)
$ErrorActionPreference = "Continue"

# Section 4: kill SELECTIVELY by command line. A blanket python kill would also
# kill the tooling that is running this harness.
$targets = Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -like "*ComfyUI*main.py*"
}
foreach ($p in $targets) {
    Write-Output ("[boot] stopping server pid {0}" -f $p.ProcessId)
    Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
}
foreach ($c in @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)) {
    Write-Output ("[boot] stopping :8000 owner pid {0}" -f $c.OwningProcess)
    Stop-Process -Id $c.OwningProcess -Force -ErrorAction SilentlyContinue
}
Start-Sleep -Seconds 6
$still = @(Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue)
if ($still.Count -gt 0) {
    Write-Output "[boot] FAILED: :8000 is still held after the reset"
    exit 2
}

if (Test-Path $LogPath) { Remove-Item $LogPath -Force -ErrorAction SilentlyContinue }
Start-Process -FilePath $Launcher -ArgumentList "`"$LogPath`""

$deadline = (Get-Date).AddSeconds($TimeoutSeconds)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if (Test-Path $LogPath) {
        $text = Get-Content $LogPath -Raw -ErrorAction SilentlyContinue
        if ($text -match 'To see the GUI go to') {
            Write-Output "[boot] server up"
            exit 0
        }
        if ($text -match 'SERVER DID NOT COME UP') {
            Write-Output "[boot] FAILED: the launcher reported the server did not come up"
            exit 3
        }
    }
}
Write-Output ("[boot] FAILED: no server after {0}s" -f $TimeoutSeconds)
exit 4
