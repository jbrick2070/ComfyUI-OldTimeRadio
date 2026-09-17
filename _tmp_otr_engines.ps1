$ErrorActionPreference = "Continue"
$oi = Invoke-RestMethod -Uri "http://127.0.0.1:8188/object_info/OTR_VideoDirector" -TimeoutSec 20
$req = $oi.OTR_VideoDirector.input.required
$opt = $oi.OTR_VideoDirector.input.optional
Write-Host "=== VideoDirector keys ==="
if ($req) { $req.PSObject.Properties.Name | ForEach-Object { Write-Host ("req {0}" -f $_) } }
if ($opt) { $opt.PSObject.Properties.Name | ForEach-Object { Write-Host ("opt {0}" -f $_) } }
$combo = $null
foreach ($bag in @($req, $opt)) {
    if ($null -eq $bag) { continue }
    foreach ($name in @("announcer_visual","character_visual","engine")) {
        if ($bag.$name) { $combo = $bag.$name; Write-Host ("combo field {0}" -f $name); break }
    }
}
# dump announcer_visual choices containing vidu/cloud
$ann = $req.announcer_visual
if (-not $ann) { $ann = $opt.announcer_visual }
if ($ann) {
    $choices = $ann[0]
    Write-Host ("announcer_visual count {0}" -f $choices.Count)
    $choices | Where-Object { $_ -match "vidu|cloud_wan|viz_camera" } | ForEach-Object { Write-Host $_ }
}
$w = Invoke-RestMethod -Uri "http://127.0.0.1:8188/object_info/OTR_LedgerScriptWriter" -TimeoutSec 20
$acts = $w.OTR_LedgerScriptWriter.input.required.act_count
if (-not $acts) { $acts = $w.OTR_LedgerScriptWriter.input.optional.act_count }
Write-Host "=== act_count spec ==="
$acts | ConvertTo-Json -Compress
Write-Host "=== Installs custom_nodes ==="
Get-ChildItem "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\custom_nodes" | Select-Object Name, Mode, LinkType | Format-Table -AutoSize
