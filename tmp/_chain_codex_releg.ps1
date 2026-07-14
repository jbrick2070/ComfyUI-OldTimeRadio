# Wait for the bake420 sweep to finish, then re-leg scifi_codex at 420w against
# the P4 audit-clamp fix (80e30c2e). ONE GPU -- this must NOT overlap the sweep.
#
# scifi_codex led the sweep and died at P4 on a 240+ char critic rationale. The
# P2 clamp fix proved out on that same leg (P0/P1/P2/P3 all landed), so this
# re-leg is the first run of the lane with BOTH clamp fixes live.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

$sweepDone = Join-Path $repo "tmp\bake420_420w_ALLDONE.txt"
$deadline = (Get-Date).AddHours(6)
while (-not (Test-Path -LiteralPath $sweepDone) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
}

"[chain] bake420 finished at $(Get-Date -Format 'HH:mm:ss'); starting scifi_codex re-leg" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg.log") -Encoding utf8

& powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File (Join-Path $repo "tmp\_sweep.ps1") `
    -Banks @("scifi_codex") -Words 420 -SweepName "bake420b"

"[chain] scifi_codex re-leg finished at $(Get-Date -Format 'HH:mm:ss')" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg.log") -Encoding utf8 -Append
