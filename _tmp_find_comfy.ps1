$ErrorActionPreference = "Continue"
Write-Host "=== python with ComfyUI in cmdline ==="
Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
    $cl = $_.CommandLine
    if ($cl -match "ComfyUI|main.py") {
        Write-Host ("pid={0}" -f $_.ProcessId)
        Write-Host $cl
        Write-Host "---"
    }
}
Write-Host "=== listeners 8000 8188 8189 ==="
@(8000, 8188, 8189, 8187) | ForEach-Object {
    $p = $_
    Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host ("port {0} pid {1}" -f $p, $_.OwningProcess) }
}
