$ErrorActionPreference = "Continue"
Set-Location "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$dir = "docs\2026-07-03-elevenlabs-tts"
# delete temp probe/launcher scripts (never commit these)
Remove-Item "$dir\_tmp_envcheck.ps1","$dir\_rt_r1.ps1","$dir\_rt_r1b.ps1","$dir\_kibitz_r2.ps1" -ErrorAction SilentlyContinue
git add "docs/2026-07-03-elevenlabs-tts" "kibitz-runs/2026-07-03-elevenlabs-tts" 2>&1 | Out-String
git commit -m "docs(cloud-audio): converged BUILD_PLAN for ElevenLabs TTS + cloud Sonilo music (roundtable R1 + kibitz r2 + Fable gate)" 2>&1 | Out-String
git push origin v2.0-alpha 2>&1 | Out-String
Write-Output "----- VERIFY -----"
git rev-parse HEAD
git rev-parse origin/v2.0-alpha
git status --short
Write-Output ("EXITCODE=" + $LASTEXITCODE)
