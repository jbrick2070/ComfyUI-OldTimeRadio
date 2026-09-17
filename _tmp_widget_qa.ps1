$ErrorActionPreference = "Continue"
$env:PYTHONUTF8 = "1"
$py = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location $repo
Write-Host "=== writer models ==="
& $py _tmp_writer_models.py
Write-Host "=== widget dump ==="
& $py _tmp_widget_diff_qa.py
Write-Host "DUMP_RC=$LASTEXITCODE"
Write-Host "=== official widget/link tests ==="
& $py -m pytest -q -p no:cacheprovider tests/test_widget_schema_order_matches_live_input_types.py tests/test_canonical_widget_input_parity.py tests/test_widget_value_alignment.py tests/test_workflow_link_target_indexes.py tests/test_shipped_template_writer_default.py tests/test_saved_workflow_model_values_resolve.py tests/test_one_act_template_stdlib.py
Write-Host "TEST_RC=$LASTEXITCODE"
Write-Host "=== build_variants --check ==="
& $py scripts/build_variants.py --check
Write-Host "VAR_RC=$LASTEXITCODE"
exit 0
