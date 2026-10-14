<#
run_meta_agent.ps1 -- headless Meta (muse-wonder) reviewer lane, driven through the Codex CLI.

WHY CODEX: the Meta Model API speaks the OpenAI Responses wire format, so `codex exec` runs it
as a custom model_provider and we inherit Codex's file-reading agent loop, its read-only
sandbox and its -o/--output-last-message file contract. This is the Meta sibling of
scripts/run_codex_agent.ps1 and scripts/run_agy_agent.ps1. The MODEL is Meta's, not OpenAI's,
so it is a different family from every driver here (Claude, Codex/GPT, Cursor/Composer) and is
a valid contrarian for any of them.

TWO THINGS THIS RUNNER DOES THAT A BARE `codex exec -p meta` DOES NOT (both measured 2026-10-13,
codex-cli 0.162.0):
  1. --ignore-user-config. The base ~/.codex/config.toml enables plugins and MCP servers whose
     tool schemas are recursive, and the Meta API refuses the whole request with
     "Recursive JSON schemas are not currently supported". A reviewer needs only the shell and
     file tools, so the lane runs on the profile alone. Auth is unaffected (the key is ours).
  2. MODEL_API_KEY is filled for THIS PROCESS ONLY from a key file, never persisted to the
     user environment. Order: an already-set $env:MODEL_API_KEY, then -KeyFile, then
     $env:OTR_META_KEY_FILE, then D:\keys\key_meta.txt.

STATUS 2026-10-13: the API leg is PROVEN (`codex exec -p meta --ignore-user-config` answered
META_OK). The repo-READING leg is NOT: the reviewer's shell commands (git, rg) were all
"rejected: blocked by policy". The cause is a codex-windows-sandbox "setup refresh had errors"
failure, and the sandbox log (~/.codex/.sandbox/sandbox.<date>.log) names
`...\runtimes\cua_node\<hash>\bin\node_repl.exe ... being used by another process (os error 32)`
-- the running Codex DESKTOP APP holds that file open. Same signature on 2026-10-08. UNPROVEN
hypothesis: with the Codex app fully closed the sandbox sets up and the reviewer can read the
repo. Free test, no model call: `codex sandbox -P ":read-only" -C <repo> -- git log --oneline -1`.
If it passes, run this script once on a trivial prompt before trusting it for a real review.

The profile is pinned in scripts/codex_profiles/meta.config.toml and copied to
$CODEX_HOME\meta.config.toml on every run, so both boxes get the same lane with no setup.

File contract (under <repo>\roundtables\runs\<RunId>\, gitignored):
  IN  : prompt_meta.md      (or pass -PromptFile to use any file)
  OUT : meta_final.md       (Meta's final message)
        meta_log.jsonl      (--json event stream + stderr)
        meta_exitcode.txt

Usage:
  powershell -ExecutionPolicy Bypass -File scripts\run_meta_agent.ps1 -RunId qa1
  powershell -ExecutionPolicy Bypass -File scripts\run_meta_agent.ps1 -RunId qa1 -PromptFile C:\tmp\review.md

Always read-only. UTF-8, no BOM.
#>
[CmdletBinding()]
param(
  [string]$Repo = "",
  [Parameter(Mandatory = $true)][string]$RunId,
  [string]$PromptFile = "",
  [ValidateSet("minimal", "low", "medium", "high", "xhigh")][string]$Reasoning = "high",
  [string]$KeyFile = ""
)

$ErrorActionPreference = "Stop"
# $PSScriptRoot is empty inside a param() default under Windows PowerShell 5.1, so the
# repo default (this script's own checkout) is resolved here instead.
if (-not $Repo) { $Repo = Split-Path $PSScriptRoot -Parent }

# --- profile: the repo copy is the authority; keep CODEX_HOME in step with it ---
$codexHome = if ($env:CODEX_HOME) { $env:CODEX_HOME } else { Join-Path $env:USERPROFILE ".codex" }
$profileSrc = Join-Path $PSScriptRoot "codex_profiles\meta.config.toml"
$profileDst = Join-Path $codexHome "meta.config.toml"
if (-not (Test-Path $profileSrc)) { throw "[meta] missing pinned profile: $profileSrc" }
New-Item -ItemType Directory -Force -Path $codexHome | Out-Null
Copy-Item $profileSrc $profileDst -Force

# --- key: process-scoped only ---
if (-not $env:MODEL_API_KEY) {
  $candidates = @($KeyFile, $env:OTR_META_KEY_FILE, "D:\keys\key_meta.txt") | Where-Object { $_ }
  $found = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $found) { throw "[meta] no API key: set MODEL_API_KEY, pass -KeyFile, or set OTR_META_KEY_FILE (tried: $($candidates -join ', '))" }
  $env:MODEL_API_KEY = (Get-Content -Raw $found).Trim()
  if (-not $env:MODEL_API_KEY) { throw "[meta] key file is empty: $found" }
}

$run = Join-Path $Repo "roundtables\runs\$RunId"
New-Item -ItemType Directory -Force -Path $run | Out-Null
$prompt = if ($PromptFile) { $PromptFile } else { Join-Path $run "prompt_meta.md" }
$out = Join-Path $run "meta_final.md"
$log = Join-Path $run "meta_log.jsonl"
$exit = Join-Path $run "meta_exitcode.txt"
if (-not (Test-Path $prompt)) { throw "[meta] missing prompt file: $prompt" }
Remove-Item $out, $exit -ErrorAction SilentlyContinue

Write-Host "[meta] run=$RunId repo=$Repo reasoning=$Reasoning sandbox=read-only -> $out" -ForegroundColor Cyan
# Under Windows PowerShell 5.1 a redirected native stderr line becomes an ErrorRecord, and with
# ErrorActionPreference=Stop the first one (codex logs tool errors there) kills the run mid-turn.
# Success is judged by the exit code and the output file, so stderr only needs to reach the log.
$ErrorActionPreference = "Continue"
Get-Content $prompt -Raw | & codex exec -p meta --ignore-user-config -C $Repo --sandbox read-only -c "model_reasoning_effort=$Reasoning" --json --color never -o $out - *> $log
$code = $LASTEXITCODE
$ErrorActionPreference = "Stop"
$code | Out-File -Encoding utf8 $exit

if ($code -ne 0) { throw "[meta] FAILED exit=$code -- see $log" }
if (-not (Test-Path $out)) { throw "[meta] exit 0 but no output file: $out (see $log)" }
Write-Host "[meta] OK exit=0 -> $out ($((Get-Item $out).Length) bytes)" -ForegroundColor Green
Get-Content $out
