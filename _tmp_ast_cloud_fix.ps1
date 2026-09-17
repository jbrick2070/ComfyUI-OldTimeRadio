$ErrorActionPreference = "Stop"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$files = @(
  "nodes/_otr_shared/cloud_media_canonical.py",
  "nodes/_otr_video_engines/eng_cloud_video.py",
  "nodes/_otr_video_engines/eng_google_veo_video.py",
  "nodes/_otr_video_engines/eng_google_omni_video.py",
  "nodes/_otr_comfy_backend.py",
  "tests/test_cloud_video_adapters.py",
  "tests/test_comfy_slot_widgets.py"
)
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
foreach ($f in $files) {
  & $py -c "import ast,sys; ast.parse(open(sys.argv[1],encoding='utf-8').read())" $f
  if ($LASTEXITCODE -ne 0) { throw "AST fail $f" }
  Write-Host "AST_OK $f"
}
Write-Host "AST_ALL_OK"
