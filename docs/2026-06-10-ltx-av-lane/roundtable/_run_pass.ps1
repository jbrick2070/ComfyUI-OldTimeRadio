# _run_pass.ps1 -- one roundtable fan-out pass for the LTX-AV lane campaign.
# Usage: powershell -NoProfile -File _run_pass.ps1 -PassNum 01 -FocusFile pass01_focus.md
# Reads pass(N-1)_plan.md (or pass00_plan.md), prepends the focus block,
# fans out to the panel, saves reviews under passNN\. ASCII, UTF-8 no BOM.
param(
    [Parameter(Mandatory=$true)][string]$PassNum,
    [Parameter(Mandatory=$true)][string]$FocusFile,
    [string]$PlanFile = ""
)
$ErrorActionPreference = 'Stop'
$py    = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$skill = 'C:\Users\jeffr\AppData\Roaming\Claude\local-agent-mode-sessions\skills-plugin\cff774c2-bb97-4b73-b186-daa560353c46\6503cb4a-a0ec-4e89-b422-c889f09084d6\skills\roundtable'
$rt    = Join-Path $skill 'scripts\roundtable_pass.py'
$cfg   = Join-Path $skill 'assets\panel.config.json'
$rp    = Join-Path $skill 'references\review-prompt.md'
$repo  = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$rtdir = Join-Path $repo 'docs\2026-06-10-ltx-av-lane\roundtable'

if (-not $PlanFile) {
    $prev = '{0:d2}' -f ([int]$PassNum - 1)
    $PlanFile = Join-Path $rtdir ("pass" + $prev + "_plan.md")
}
$focus = [System.IO.File]::ReadAllText((Join-Path $rtdir $FocusFile))
$plan  = [System.IO.File]::ReadAllText($PlanFile)
$inp   = Join-Path $rtdir ("pass" + $PassNum + "_input.md")
$enc   = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($inp, $focus + "`n`n" + $plan, $enc)

$glist = Join-Path $rtdir ("pass" + $PassNum + "_grounding.txt")
if (Test-Path $glist) {
    $ground = Get-Content $glist | Where-Object { $_.Trim() -and -not $_.StartsWith('#') } |
        ForEach-Object { Join-Path $repo $_.Trim() }
} else {
    $ground = @(
        (Join-Path $repo 'nodes\_otr_video_engines\registry.py'),
        (Join-Path $repo 'nodes\_otr_video_engines\eng_humo.py'),
        (Join-Path $repo 'nodes\_otr_video_engines\eng_ltx_video.py'),
        (Join-Path $repo 'nodes\_otr_video_engines\motion_common.py'),
        (Join-Path $repo 'nodes\_otr_shared\role_compat.py'),
        (Join-Path $repo 'nodes\_otr_shared\fallback.py'),
        (Join-Path $repo 'nodes\otr_video_director.py'),
        (Join-Path $repo 'nodes\_otr_video_engines\schemas.py')
    )
}
$out = Join-Path $rtdir ("pass" + $PassNum)
New-Item -ItemType Directory -Force -Path $out | Out-Null
& $py $rt --doc $inp --out $out --config $cfg --prompt $rp `
    --grounding @ground `
    --models '~openai/gpt-latest,~google/gemini-pro-latest,deepseek/deepseek-v4-pro' `
    --max-tokens 12000 --reasoning-effort none --temperature 0.5 `
    --budget 3.00 *>&1 | Out-File -FilePath (Join-Path $out 'run.log') -Encoding utf8
Write-Host ("PASS " + $PassNum + " DONE rc=" + $LASTEXITCODE)
