<#
run_codex_agent.ps1 -- robust headless Codex runner (file-contract; no stdout scraping).

PROVEN PATTERN (codex-cli 0.142.3, 2026-06-27): Codex writes its FINAL answer to a known
file via -o/--output-last-message; success is judged by the EXIT CODE + that file existing,
NEVER by scraping terminal text. The prompt is piped in via stdin (`-`). JSONL events go to a
log. This replaces the fragile "capture stdout" path in the roundtable-local skill's
agent_roundtable.py.

File contract (under <repo>\roundtables\runs\<RunId>\):
  IN  : prompt_codex.md
  OUT : codex_final.md        (Codex's final message)
        codex_log.jsonl       (--json event stream + stderr)
        codex_exitcode.txt    (subprocess exit code)

Modes (least privilege first):
  review (default)  -> --sandbox read-only        (CANNOT edit; correct for reviews)
  edit              -> --full-auto                (workspace-write; for delegated edits)
  edit -Dangerous   -> --dangerously-bypass-approvals-and-sandbox  (FULL bypass; ISOLATED
                       worktree/clone ONLY -- guarded below; never the real home dir)

Usage:
  pwsh run_codex_agent.ps1 -RunId r4 -Mode review
  pwsh run_codex_agent.ps1 -RunId wire1 -Mode edit            # full-auto, workspace-write
  pwsh run_codex_agent.ps1 -RunId wire1 -Mode edit -Dangerous # bypass; requires -Repo a worktree

Do NOT launch two full-permission (edit) Codex jobs in the same repo at once -- use separate
git worktrees. Reviews (read-only) are safe to parallelize. UTF-8, no BOM.
#>
[CmdletBinding()]
param(
  [string]$Repo = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio",
  [Parameter(Mandatory = $true)][string]$RunId,
  [ValidateSet("review", "edit")][string]$Mode = "review",
  [switch]$Dangerous,
  [string]$CodexBin = "C:\Users\jeffr\AppData\Local\OpenAI\Codex\bin\aec6b7c6fcdfb66a"
)

$ErrorActionPreference = "Stop"
if (Test-Path $CodexBin) { $env:PATH = "$CodexBin;$env:PATH" }

$run = Join-Path $Repo "roundtables\runs\$RunId"
New-Item -ItemType Directory -Force -Path $run | Out-Null
$prompt = Join-Path $run "prompt_codex.md"
$out = Join-Path $run "codex_final.md"
$log = Join-Path $run "codex_log.jsonl"
$exit = Join-Path $run "codex_exitcode.txt"

if (-not (Test-Path $prompt)) { throw "missing prompt file: $prompt" }
Remove-Item $out, $exit -ErrorAction SilentlyContinue

# pick the sandbox flag by mode (least privilege)
switch ($Mode) {
  "review" { $sandboxArgs = @("--sandbox", "read-only") }
  "edit" {
    if ($Dangerous) {
      # FULL bypass -- only inside an isolated worktree/clone, never the real home dir.
      if ($Repo -notmatch "worktree|wt-|\.tmp|clone") {
        throw "Refusing --dangerously-bypass-approvals-and-sandbox on $Repo : pass -Repo a git worktree/clone (name must contain 'worktree','wt-','.tmp', or 'clone')."
      }
      $sandboxArgs = @("--dangerously-bypass-approvals-and-sandbox")
    }
    else { $sandboxArgs = @("--full-auto") }
  }
}

Write-Host "[codex] run=$RunId mode=$Mode repo=$Repo" -ForegroundColor Cyan
Write-Host "[codex] flags: $($sandboxArgs -join ' ') --json --color never -o codex_final.md" -ForegroundColor DarkGray

# robust contract: prompt via stdin (`-`); final answer to a FILE; events to JSONL.
Get-Content $prompt -Raw | & codex exec -C $Repo @sandboxArgs --json --color never -o $out - *> $log
$code = $LASTEXITCODE
$code | Out-File -Encoding utf8 $exit

if ($code -ne 0) { throw "[codex] FAILED exit=$code -- see $log" }
if (-not (Test-Path $out)) { throw "[codex] exit 0 but no output file: $out (see $log)" }
Write-Host "[codex] OK exit=0 -> $out ($((Get-Item $out).Length) bytes)" -ForegroundColor Green
Get-Content $out
