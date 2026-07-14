# Kill ONLY the armed re-leg chain (it holds a stale one-bank list in memory),
# then relaunch it with the current two-bank list. Selective by CommandLine --
# a blanket powershell/python kill would sever the MCP tools and the live sweep.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

$victims = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*_chain_codex_releg.ps1*" }

foreach ($v in $victims) {
    "killing stale re-leg chain pid=$($v.ProcessId)"
    Stop-Process -Id $v.ProcessId -Force -ErrorAction SilentlyContinue
}
if (-not $victims) { "no armed re-leg chain found" }

Start-Sleep -Seconds 1
Start-Process -FilePath "powershell.exe" -ArgumentList `
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", `
    (Join-Path $repo "tmp\_chain_codex_releg.ps1") -WindowStyle Hidden

"re-leg chain re-armed with: scifi_codex, public_domain_story"

# Confirm the 720 gate is still armed (it waits on bake420b, which now covers both).
$gate = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*_chain_720.ps1*" }
if ($gate) { "720 gate still armed (pid=$($gate.ProcessId))" }
else { "WARNING: 720 gate NOT armed" }
