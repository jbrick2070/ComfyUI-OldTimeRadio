$ErrorActionPreference = "Continue"
$pidListen = 42400
$p = Get-CimInstance Win32_Process -Filter "ProcessId=42400"
Write-Host ("name={0}" -f $p.Name)
if ($p.CommandLine) {
    Write-Host $p.CommandLine.Substring(0, [Math]::Min(500, $p.CommandLine.Length))
}
Write-Host "=== OTR on 8188 ==="
foreach ($node in @("OTR_LedgerScriptWriter", "OTR_VideoDirector", "OTR_OBSPublish")) {
    try {
        $r = Invoke-WebRequest -Uri ("http://127.0.0.1:8188/object_info/{0}" -f $node) -UseBasicParsing -TimeoutSec 8
        Write-Host ("OK {0} bytes={1}" -f $node, $r.Content.Length)
    } catch {
        Write-Host ("MISS {0} {1}" -f $node, $_.Exception.Message)
    }
}
Write-Host "=== system_stats ==="
try {
    $s = Invoke-WebRequest -Uri "http://127.0.0.1:8188/system_stats" -UseBasicParsing -TimeoutSec 5
    Write-Host $s.Content.Substring(0, [Math]::Min(800, $s.Content.Length))
} catch {
    Write-Host $_.Exception.Message
}
Write-Host "=== /extensions sample ==="
try {
    $e = Invoke-WebRequest -Uri "http://127.0.0.1:8188/extensions" -UseBasicParsing -TimeoutSec 5
    Write-Host $e.Content.Substring(0, [Math]::Min(400, $e.Content.Length))
} catch {
    Write-Host "no extensions"
}
