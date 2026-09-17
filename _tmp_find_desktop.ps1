$ErrorActionPreference = "Continue"
Write-Host "=== Comfy Desktop processes ==="
Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "Comfy|electron" -and $_.CommandLine -match "Comfy"
} | ForEach-Object {
    Write-Host ("pid={0} name={1}" -f $_.ProcessId, $_.Name)
    Write-Host $_.CommandLine
    Write-Host "---"
}
Get-ChildItem "C:\Users\jeffr\AppData\Local\Programs" -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "Comfy" }
Get-ChildItem "C:\Users\jeffr\AppData\Local" -Directory -ErrorAction SilentlyContinue | Where-Object { $_.Name -match "Comfy" }
