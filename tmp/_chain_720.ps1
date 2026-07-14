# GATE: the 720 sweep runs only when 420 is GREEN ON ALL 10 BANKS.
#
# Nobody skips the ladder (operator, explicitly: the 420 rung comes before 720).
# If a bank is still red at 420, this does NOT launch 720 -- it writes
# tmp\bake720_GATE_BLOCKED.txt naming the offenders and stops. A red bank needs a
# root fix and a re-leg, not a promotion to a longer, more expensive rung.
$repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
Set-Location -LiteralPath $repo

$banks = @(
    "scifi_codex", "shakespeare", "scifi_sonnet", "original_codex56sol",
    "scifi_gemini", "public_domain_story", "scifi_fable2", "media_archive",
    "science_news", "original_radio"
)

# Wait for the codex re-leg chain (which itself waits for the main sweep).
$relegDone = Join-Path $repo "tmp\bake420b_420w_ALLDONE.txt"
$deadline = (Get-Date).AddHours(9)
while (-not (Test-Path -LiteralPath $relegDone) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
}

# Latest verdict per bank at 420, across BOTH the main sweep and the re-leg
# sweep. A later re-leg supersedes an earlier failure for the same bank.
$verdict = @{}
foreach ($name in @("bake420", "bake420b")) {
    $sum = Join-Path $repo ("tmp\{0}_420w_summary.txt" -f $name)
    if (-not (Test-Path -LiteralPath $sum)) { continue }
    foreach ($line in (Get-Content -LiteralPath $sum)) {
        if ($line -match "END\s+(\S+)\s+420w\s+::\s+EXIT=\d+\s+::.*RESULT (\w+)") {
            $verdict[$Matches[1]] = $Matches[2]
        }
    }
}

$red = @()
foreach ($b in $banks) {
    if ($verdict[$b] -ne "SUCCESS") {
        $red += ("{0}={1}" -f $b, $(if ($verdict[$b]) { $verdict[$b] } else { "NO_RESULT" }))
    }
}

if ($red.Count -gt 0) {
    $msg = @(
        "420 GATE BLOCKED at $(Get-Date -Format 'HH:mm:ss'). 720 NOT launched.",
        "Red banks: " + ($red -join ", "),
        "Fix at the root, re-leg the bank at 420, then re-arm tmp\_chain_720.ps1."
    ) -join [Environment]::NewLine
    $msg | Out-File -FilePath (Join-Path $repo "tmp\bake720_GATE_BLOCKED.txt") -Encoding utf8
    exit 1
}

"420 GREEN ON ALL 10 at $(Get-Date -Format 'HH:mm:ss'); launching the 720 sweep." |
    Out-File -FilePath (Join-Path $repo "tmp\bake720_GATE_OPEN.txt") -Encoding utf8

& powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File (Join-Path $repo "tmp\_sweep.ps1") `
    -Banks $banks -Words 720 -SweepName "bake720"
