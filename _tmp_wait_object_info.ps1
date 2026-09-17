$ErrorActionPreference = "Continue"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$deadline = (Get-Date).AddMinutes(2)
do {
    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter" -UseBasicParsing -TimeoutSec 5
        $j = $r.Content | ConvertFrom-Json
        $info = $j.OTR_LedgerScriptWriter
        if (-not $info) { $info = $j.PSObject.Properties.Value | Select-Object -First 1 }
        $req = @($info.input.required.PSObject.Properties.Name)
        $opt = @($info.input.optional.PSObject.Properties.Name)
        $all = $req + $opt
        Write-Host ("REQUIRED=" + ($req -join ","))
        Write-Host ("COUNT=" + $all.Count)
        Write-Host ("spacesaver=" + ($all -contains "perfect_run_spacesaver"))
        Write-Host ("creative_in=" + ($all -contains "creative_writing_model"))
        exit 0
    } catch {
        Write-Host ("wait: " + $_.Exception.Message)
        Start-Sleep -Seconds 5
    }
} while ((Get-Date) -lt $deadline)
Write-Host "8188 not up yet"
exit 1
