# Re-arm the 720 gate (it holds a stale wait-list in memory) and launch the
# bake420d chain. Runs from a FILE so its own command line cannot match the
# kill filter -- doing this inline once killed the very shell running it.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"

$self = $PID
$victims = Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object {
        $_.ProcessId -ne $self -and
        $_.CommandLine -like "*-File*_chain_720.ps1*"
    }
foreach ($v in $victims) {
    "killing stale 720 gate pid=$($v.ProcessId)"
    Stop-Process -Id $v.ProcessId -Force -ErrorAction SilentlyContinue
}
if (-not $victims) { "no stale 720 gate found" }

Start-Sleep -Seconds 1

Start-Process -FilePath "powershell.exe" -ArgumentList `
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", `
    (Join-Path $repo "tmp\_chain_releg_d.ps1") -WindowStyle Hidden
Start-Process -FilePath "powershell.exe" -ArgumentList `
    "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", `
    (Join-Path $repo "tmp\_chain_720.ps1") -WindowStyle Hidden

Start-Sleep -Seconds 2
"=== chains armed ==="
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" |
    Where-Object { $_.CommandLine -like "*-File*tmp\*" } |
    ForEach-Object { "  pid=" + $_.ProcessId + " :: " + (Split-Path $_.CommandLine -Leaf) }
