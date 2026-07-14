# Wait for the bake420 sweep to finish, then RE-LEG every bank that died, against
# the fixes shipped during the sweep. ONE GPU -- this must NOT overlap the sweep.
#
#   scifi_codex          -- died at P4 (an audit dying on the length of its own
#                           240-char rationale). Fixed 80e30c2e. Its P2 clamp fix
#                           already proved out on that same leg (P0-P3 all landed).
#   public_domain_story  -- died at the announcer news-coda bridge: aion-3.0-mini
#                           is a MANDATORY-reasoning model, `max_tokens` bounds
#                           reasoning + content TOGETHER with the reasoning first,
#                           and the 1024 floor was a thinking budget with no room
#                           left to answer. finish_reason=length on both attempts.
#                           Floor is now reasoning-aware.
#
# Banks are appended here as they fail, so this stays the single re-leg queue.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

#   scifi_fable2         -- died at the 'critic' pass: CriticNote.speaker was
#                           REQUIRED, but empty is a legal value meaning "this note
#                           is about the scene, not one character" (its own consumer
#                           `_note_scope_error` says so on both branches). The critic
#                           wrote a scene-wide note, omitted the key, and died.
#                           speaker now defaults to "".
$relegBanks = @("scifi_codex", "public_domain_story", "scifi_fable2")

$sweepDone = Join-Path $repo "tmp\bake420_420w_ALLDONE.txt"
$deadline = (Get-Date).AddHours(8)
while (-not (Test-Path -LiteralPath $sweepDone) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
}

"[chain] bake420 finished at $(Get-Date -Format 'HH:mm:ss'); re-legging: $($relegBanks -join ', ')" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg.log") -Encoding utf8

& powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File (Join-Path $repo "tmp\_sweep.ps1") `
    -Banks $relegBanks -Words 420 -SweepName "bake420b"

"[chain] re-legs finished at $(Get-Date -Format 'HH:mm:ss')" |
    Out-File -FilePath (Join-Path $repo "tmp\chain_releg.log") -Encoding utf8 -Append
