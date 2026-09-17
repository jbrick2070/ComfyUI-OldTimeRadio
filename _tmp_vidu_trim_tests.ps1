$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
& $py -m pytest -q -p no:cacheprovider `
  tests/test_cloud_video_adapters.py `
  tests/test_comfy_slot_widgets.py `
  tests/test_frame_receipt_conformance.py `
  tests/test_word_razzle.py `
  tests/test_generation_budget.py `
  tests/test_engine_contract_roster.py
Write-Host "PYTEST_RC=$LASTEXITCODE"
Write-Host "=== 8188 ==="
try {
    $q = Invoke-RestMethod -Uri "http://127.0.0.1:8188/queue" -TimeoutSec 5
    $id = if (@($q.queue_running).Count) { $q.queue_running[0][1] } else { "-" }
    Write-Host ("running={0} pending={1} id={2}" -f @($q.queue_running).Count, @($q.queue_pending).Count, $id)
} catch { Write-Host "8188 FAIL" }
