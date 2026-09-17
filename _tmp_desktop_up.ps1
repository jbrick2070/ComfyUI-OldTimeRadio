$ErrorActionPreference = "Continue"
Write-Host "=== Comfy Desktop ==="
Get-Process -Name "Comfy Desktop" -ErrorAction SilentlyContinue | Select-Object Id, StartTime
Write-Host "=== python Comfy ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object {
    $_.CommandLine -match "ComfyUI"
} | ForEach-Object {
    Write-Host ("pid={0}" -f $_.ProcessId)
    Write-Host $_.CommandLine
}
Write-Host "=== listen 8188/8000 ==="
@(8188, 8000, 8189) | ForEach-Object {
    Get-NetTCPConnection -LocalPort $_ -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host ("port {0} pid {1}" -f $_.LocalPort, $_.OwningProcess) }
}
Write-Host "=== junction ==="
cmd /c dir "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes"
