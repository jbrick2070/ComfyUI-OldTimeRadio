# Rotate one log file out of the way so a fresh boot cannot truncate it.
#
# WHY THIS EXISTS (2026-08-04): `_otr_soak_server_launch.cmd` redirected with
# `>`, so every reboot destroyed the previous run's server log. A 320-word leg
# failed at 22:11 with two stills unmaterialized; the 23:06 relaunch truncated
# the only log that could have said why. The evidence was gone before anyone
# looked for it.
#
# It lives in its own script, rather than inline in the .cmd, because the
# rotation needs a locale-independent timestamp and path surgery -- exactly the
# nested-quoting shape that cmd mangles.
#
# NOTE THE NAME: no leading underscore. `.gitignore` ignores `scripts/_*.ps1`
# as scratch, and this helper is LOAD-BEARING -- the launcher calls it on every
# boot, so an ignored copy would work on this box and be missing everywhere
# else.
#
# CONTRACT: the caller's log path is NEVER repointed. This only MOVES an
# existing file aside; the caller then writes a fresh log at the original path.
# The live harnesses that depend on that path: otr_headless_canonical.ps1,
# _otr_w45_boot.ps1, otr_ia2v_server_boot.cmd. (The bakeoff runners and the
# HuMo VRAM ladder that used to be in this roster were all retired.)
#
# EXIT 0 = nothing to rotate, or rotated successfully.
# EXIT 1 = a prior log exists and could NOT be moved. The caller must then
#          APPEND rather than truncate, so a locked log is still not destroyed.

param([Parameter(Mandatory = $true)][string]$Path)

$ErrorActionPreference = 'Stop'

try {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        exit 0   # nothing to preserve
    }
    $dir  = [System.IO.Path]::GetDirectoryName($Path)
    $stem = [System.IO.Path]::GetFileNameWithoutExtension($Path)
    $ext  = [System.IO.Path]::GetExtension($Path)
    if ([string]::IsNullOrEmpty($ext)) { $ext = '.log' }
    # A bare filename ("server.log") has no directory component, and
    # Join-Path THROWS on an empty -Path -- which would make rotation fail
    # forever for that caller shape instead of just once. Resolve it against
    # the current directory, the same place the caller's own relative path
    # would land.
    if ([string]::IsNullOrEmpty($dir)) { $dir = (Get-Location).Path }

    # Sortable, locale-independent, millisecond resolution. %DATE%/%TIME% are
    # locale-dependent and contain '/' and ':' on many locales, which are
    # invalid in a filename -- that is why the stamp is minted here.
    $stamp  = (Get-Date).ToString('yyyyMMdd_HHmmss_fff')
    $target = Join-Path $dir ($stem + '.' + $stamp + $ext)

    # Millisecond collision is unlikely but not impossible when two harnesses
    # boot together; never silently overwrite a rotated log.
    $n = 1
    while (Test-Path -LiteralPath $target) {
        $target = Join-Path $dir ($stem + '.' + $stamp + '_' + $n + $ext)
        $n++
        if ($n -gt 50) { exit 1 }
    }

    Move-Item -LiteralPath $Path -Destination $target -ErrorAction Stop
    Write-Output ("rotated -> " + [System.IO.Path]::GetFileName($target))
    exit 0
}
catch {
    # A locked file (a server still holding it) lands here. Report failure so
    # the caller appends instead of truncating.
    Write-Output ("rotation failed: " + $_.Exception.Message)
    exit 1
}
