$ErrorActionPreference = "Stop"
$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Import-Module -Name (Join-Path $Repo "scripts\otr_headless_process.psm1") -Force
$procs = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
foreach ($proc in $procs) {
    $cmd = [string]$proc.CommandLine
    if (-not $cmd) { continue }
    if ((Test-OtrHeadlessServerCommand -CommandLine $cmd) -or (Test-OtrCanonicalRunnerCommand -CommandLine $cmd)) {
        Write-Output ("stopping pid={0}" -f $proc.ProcessId)
        Stop-Process -Id $proc.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
# also stop the foley_mystory launcher powershell if it is still waiting on the runner
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" -ErrorAction SilentlyContinue | ForEach-Object {
    $cmd = [string]$_.CommandLine
    if ($cmd -match '_tmp_foley_mystory') {
        Write-Output ("stopping launcher pid={0}" -f $_.ProcessId)
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
}
Start-Sleep -Seconds 2
$listen = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($listen) { Write-Output "port 8000 still listening" } else { Write-Output "port 8000 clear" }
