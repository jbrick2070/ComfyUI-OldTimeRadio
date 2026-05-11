$ErrorActionPreference = 'SilentlyContinue'
$out = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\model_inventory.txt'
$roots = @(
  'C:\Users\jeffr\Documents\ComfyUI\models',
  'C:\Users\jeffr\.cache\huggingface\hub',
  'C:\Users\jeffr\AppData\Local\Programs\ComfyUI\resources\ComfyUI\models'
)
$exts = @('*.safetensors','*.ckpt','*.gguf','*.bin','*.pth','*.pt','*.onnx')
'Scanning start: ' + (Get-Date -Format 'yyyy-MM-dd HH:mm:ss') | Out-File -FilePath $out -Encoding utf8
$all = foreach($r in $roots){
  if(Test-Path $r){
    ('--- root: ' + $r + ' ---') | Out-File -FilePath $out -Append -Encoding utf8
    Get-ChildItem -Path $r -Recurse -File -Include $exts -Force -ErrorAction SilentlyContinue |
      Where-Object { $_.Length -gt 100MB }
  }
}
'--- results ---' | Out-File -FilePath $out -Append -Encoding utf8
$all |
  Sort-Object Length -Descending |
  Select-Object @{n='GB';e={[math]::Round($_.Length/1GB,2)}},@{n='Path';e={$_.FullName}} |
  Format-Table -AutoSize |
  Out-String -Width 260 |
  Out-File -FilePath $out -Append -Encoding utf8
('Total matches: ' + $all.Count) | Out-File -FilePath $out -Append -Encoding utf8
('Total bytes:   ' + ($all | Measure-Object Length -Sum).Sum) | Out-File -FilePath $out -Append -Encoding utf8
'DONE' | Out-File -FilePath $out -Append -Encoding utf8
