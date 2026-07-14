# Wait for the bake420b re-legs to drain, then RE-LEG scifi_codex once more --
# this time against the OTR_MasterAudioMux fixes.
#
# bake420b's scifi_codex was a PHANTOM: RESULT SUCCESS, no episode. The terminal
# publish node caught its own fail-closed ValueError and returned an empty path, so
# a 25-minute render that never published was recorded green. Both defects are now
# fixed:
#   - the publish node RAISES instead of swallowing (a failed publish fails the run)
#   - the duration gate absorbs concat frame quantization (it refused to publish a
#     finished episode over ~0.05s, about 1.25 frames)
#
# GPU is serialized: this must not overlap bake420b.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

$bDone = Join-Path $repo "tmp\bake420b_420w_ALLDONE.txt"
$deadline = (Get-Date).AddHours(4)
while (-not (Test-Path -LiteralPath $bDone) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 20
}

"[chain-c] bake420b drained at $(Get-Date -Format 'HH:mm:ss'); re-legging scifi_codex vs the mux fix" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg_c.log") -Encoding utf8

# IN-PROCESS call with a REAL array -- `powershell -File` cannot carry an array.
& (Join-Path $repo "tmp\_sweep.ps1") -Banks @("scifi_codex") -Words 420 -SweepName "bake420c"

"[chain-c] done at $(Get-Date -Format 'HH:mm:ss')" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg_c.log") -Encoding utf8 -Append
