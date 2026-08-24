# Selective server reset for scripts/otr_overnight_loop.sh, in its own file
# on purpose (2026-08-24). The loop used to build this as a nested
# powershell -NoProfile -Command "..." STRING assembled inside a bash
# double-quoted string (backslash-escaped quotes, an escaped backtick to
# embed a literal PowerShell `" inside it). That string was syntactically
# correct PowerShell -- it matched the documented working
# Start-Process -FilePath ... -ArgumentList "`"$LOG`"" recipe exactly -- and
# still failed silently every time the loop's OWN health check invoked it:
# git-bash's MSYS argument translation re-quotes a string handed to a native
# Windows exe (like powershell.exe) before CreateProcess ever sees it, which
# is a different quoting pass than a human typing the same command straight
# into a PowerShell prompt. CLAUDE.md already names the fix for exactly this
# shape of bug: move anything that would need nested quotes into its own
# script file instead of iterating on escaping. This file is that move.
#
# Called with two plain string arguments -- no bash-side quoting to get
# wrong, because there is none left to get wrong.
param(
    [Parameter(Mandatory = $true)][string]$LauncherCmd,
    [Parameter(Mandatory = $true)][string]$BootLog
)

Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object {
        $_.CommandLine -match 'ComfyUI-OldTimeRadio|main\.py.*--port 8000|otr_writer_bank_gate|otr_canonical_api_run'
    } |
    ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }

Start-Sleep -Seconds 5

$listening = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if (-not $listening) {
    Start-Process -FilePath $LauncherCmd -ArgumentList "`"$BootLog`"" -WindowStyle Hidden
}
