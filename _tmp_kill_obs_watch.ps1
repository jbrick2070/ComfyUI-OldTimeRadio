$ErrorActionPreference = "Continue"
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "otr_pod_obs_bridge"
} | ForEach-Object {
    Write-Host ("KILL python pid={0}" -f $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force
}
Get-CimInstance Win32_Process -Filter "Name='powershell.exe'" | Where-Object {
    $_.CommandLine -match "_tmp_pod_obs_watch"
} | ForEach-Object {
    Write-Host ("KILL powershell pid={0}" -f $_.ProcessId)
    Stop-Process -Id $_.ProcessId -Force
}
Start-Sleep -Seconds 2
Write-Host "remaining bridge:"
Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match "otr_pod_obs_bridge|_tmp_pod_obs_watch"
} | ForEach-Object { Write-Host $_.ProcessId $_.Name }
exit 0
