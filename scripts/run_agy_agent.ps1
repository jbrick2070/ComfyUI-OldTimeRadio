<#
run_agy_agent.ps1 -- robust headless AntiGravity (agy) runner via the FILE-HANDOFF protocol.

WHY: agy's `--print`/`-p` mode swallows stdout when redirected on Windows (needs a real TTY),
so capturing its output fails (exit 0, empty). FIX (proven 2026-06-27, agy 1.0.13): agy is an
AGENT, so we instruct it IN THE PROMPT to WRITE its final review to a file using its own write
tool, then read that file. Stdout is never relied on. Success = exit 0 AND the output file exists.

This is the agy sibling of scripts/run_codex_agent.ps1 (which uses Codex's native
-o/--output-last-message). Together they give the /roundtable-local skill a real 2-agent local
panel on this machine.

SAFETY: agy has no read-only-that-still-writes-the-output sandbox, so the file write needs
`--dangerously-skip-permissions` (auto-approve tool calls = UNSANDBOXED agent). We mitigate:
  - a strict prompt directive (review-only; write ONE file; modify nothing else),
  - the repo is git-committed so any stray edit shows in `git status` and is revertible,
  - optional -Worktree creates a throwaway git worktree so agy physically cannot touch your real
    working tree (hard isolation; recommended for untrusted prompts).

File contract (under <repo>\roundtables\runs\<RunId>\):
  IN  : prompt_agy.md      (the review prompt; the output directive is appended automatically)
  OUT : agy_final.md       (agy's review, written by agy itself)
        agy_log.txt        (agy stdout+stderr -- usually empty; diagnostic only)
        agy_exitcode.txt

Usage:
  pwsh run_agy_agent.ps1 -RunId r4
  pwsh run_agy_agent.ps1 -RunId r4 -Worktree     # hard-isolated copy at HEAD

UTF-8, no BOM.
#>
[CmdletBinding()]
param(
  [string]$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio",
  [Parameter(Mandatory = $true)][string]$RunId,
  [switch]$Worktree,
  [string]$AgyBin = "C:\Users\jeffr\AppData\Local\agy\bin"
)

$ErrorActionPreference = "Stop"
if (Test-Path $AgyBin) { $env:PATH = "$AgyBin;$env:PATH" }

$run = Join-Path $Repo "roundtables\runs\$RunId"
New-Item -ItemType Directory -Force -Path $run | Out-Null
$promptIn = Join-Path $run "prompt_agy.md"
$out = Join-Path $run "agy_final.md"
$log = Join-Path $run "agy_log.txt"
$exit = Join-Path $run "agy_exitcode.txt"
if (-not (Test-Path $promptIn)) { throw "missing prompt file: $promptIn" }
Remove-Item $out, $exit -ErrorAction SilentlyContinue

# the work dir agy reads in: a throwaway worktree (hard isolation) or the repo itself.
$workdir = $Repo
$wt = $null
if ($Worktree) {
  $wt = Join-Path $env:TEMP ("agy_wt_" + $RunId + "_" + (Get-Date -Format "HHmmss"))
  Write-Host "[agy] creating isolated worktree at $wt" -ForegroundColor DarkGray
  git -C $Repo worktree add --detach $wt HEAD | Out-Null
  $workdir = $wt
  $out = Join-Path $wt "agy_final.md"   # agy writes inside its own workdir
}

# strict output directive appended to the review prompt (the file-handoff).
$directive = @"

------------------------------------------------------------------
OUTPUT CONTRACT (MANDATORY): You are a READ-ONLY reviewer. Do NOT modify, create, or delete any
file EXCEPT the single output file below. Write your COMPLETE review to this exact path, then stop:
  $out
Write nothing to stdout. After writing the file, exit immediately.
"@
$prompt = (Get-Content $promptIn -Raw) + $directive

Write-Host "[agy] run=$RunId workdir=$workdir -> $out" -ForegroundColor Cyan
Push-Location $workdir
try {
  & agy --dangerously-skip-permissions -p $prompt 1> $log 2>&1
  $code = $LASTEXITCODE
}
finally { Pop-Location }
$code | Out-File -Encoding utf8 $exit

# if isolated, copy the review back into the real run dir
if ($Worktree -and (Test-Path $out)) {
  Copy-Item $out (Join-Path $run "agy_final.md") -Force
  $out = Join-Path $run "agy_final.md"
}
if ($wt) { git -C $Repo worktree remove --force $wt 2>$null | Out-Null }

if ($code -ne 0) { throw "[agy] FAILED exit=$code -- see $log" }
if (-not (Test-Path $out)) { throw "[agy] exit 0 but no output file: $out (agy ignored the write directive; see $log)" }
Write-Host "[agy] OK exit=0 -> $out ($((Get-Item $out).Length) bytes)" -ForegroundColor Green
Get-Content $out
