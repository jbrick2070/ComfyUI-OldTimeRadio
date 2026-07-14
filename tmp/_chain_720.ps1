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

# Wait for EVERY re-leg sweep to drain (bake420b, then bake420c -- the scifi_codex
# re-run against the MasterAudioMux phantom-green fix).
$deadline = (Get-Date).AddHours(9)
foreach ($doneFile in @("tmp\bake420b_420w_ALLDONE.txt", "tmp\bake420c_420w_ALLDONE.txt")) {
    $p = Join-Path $repo $doneFile
    while (-not (Test-Path -LiteralPath $p) -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 30
    }
}

# Latest verdict per bank at 420, across the main sweep AND every re-leg sweep.
# A later re-leg supersedes an earlier failure for the same bank.
$verdict = @{}
foreach ($name in @("bake420", "bake420b", "bake420c")) {
    $sum = Join-Path $repo ("tmp\{0}_420w_summary.txt" -f $name)
    if (-not (Test-Path -LiteralPath $sum)) { continue }
    foreach ($line in (Get-Content -LiteralPath $sum)) {
        if ($line -match "END\s+(\S+)\s+420w\s+::\s+EXIT=\d+\s+::.*RESULT (\w+)") {
            $verdict[$Matches[1]] = $Matches[2]
        }
    }
}

# RESULT SUCCESS IS NOT PROOF -- demand the ASSET. OTR_MasterAudioMux used to
# swallow its own fail-closed errors and return an empty path, so a leg could
# report SUCCESS with no episode on disk (live 2026-07-14, scifi_codex 5ab3884b).
# The gate must never promote a PHANTOM to the 720 rung. Ground truth is
# `obs_publish OK` in that leg's server log.
function Test-Published([string]$bank) {
    $legLog = Join-Path $repo ("tmp\leg_{0}_420w.log" -f $bank)
    if (-not (Test-Path -LiteralPath $legLog)) { return $false }
    $portHit = Select-String -Path $legLog -Pattern "port=(\d+)" | Select-Object -Last 1
    if (-not $portHit) { return $false }
    $srv = Join-Path $repo ("tmp\otr_headless_{0}.log" -f $portHit.Matches[0].Groups[1].Value)
    if (-not (Test-Path -LiteralPath $srv)) { return $false }
    return [bool](Select-String -Path $srv -Pattern "obs_publish OK" -Quiet)
}

$red = @()
foreach ($b in $banks) {
    if ($verdict[$b] -ne "SUCCESS") {
        $red += ("{0}={1}" -f $b, $(if ($verdict[$b]) { $verdict[$b] } else { "NO_RESULT" }))
    }
    elseif (-not (Test-Published $b)) {
        $red += ("{0}=PHANTOM_NO_ASSET" -f $b)
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
