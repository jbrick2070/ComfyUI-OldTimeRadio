$ErrorActionPreference = "Stop"
$env:KIBITZ_AGY_PRINT_TIMEOUT = "15m"
$agy = "C:\Users\jeffr\AppData\Local\agy\bin\agy.EXE"
$promptFile = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\_tmp_agy_qa_prompt.txt"
$out = "C:\Users\jeffr\AppData\Local\Temp\otr_qa_ab019f54_agy.md"
if (Test-Path $out) { Remove-Item $out -Force }
# agy -p= must keep the prompt on the flag. Do not split -p from its value.
& $agy models
$p = "Read $promptFile and follow it exactly. Write the complete review only to $out then stop. Do not edit the repo."
& $agy --dangerously-skip-permissions --print-timeout 15m -p="$p"
Write-Output ("AGY_RC=" + $LASTEXITCODE)
if (Test-Path $out) {
    Write-Output ("AGY_OUT_BYTES=" + (Get-Item $out).Length)
} else {
    Write-Output "AGY_OUT_MISSING"
}
exit $LASTEXITCODE
