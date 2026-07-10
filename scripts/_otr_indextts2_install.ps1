# OTR IndexTTS2 Path-B install -- the isolated-venv sidecar for the DEFAULT
# char_voice engine. This is the script named by eng_indextts2.py's
# fail-closed "Path B not installed" error. Mirrors _otr_chatterbox_install.ps1
# in spirit: index-tts pins its OWN stack (python 3.10 + torch 2.8 -- cu128 on
# Windows/Linux, Metal-capable default wheels on Mac, selected automatically by
# the repo's [tool.uv.sources]) so it NEVER touches the main ComfyUI venv.
#
# Steps:
#   1) bootstrap uv if missing (the repo's official installer)
#   2) clone index-tts into <ComfyUI>\index-tts and pin the TESTED commit
#   3) uv sync  (creates .venv from .python-version + uv.lock; installs the
#      project editable so `import indextts` resolves in the worker)
#   4) download + validate weights via scripts\_otr_idx_download_weights.py
#   5) readiness smoke via scripts\_otr_indextts2_worker.py
#
# Re-run safe: every step is idempotent.
$ErrorActionPreference = "Stop"

# <repo>\scripts -> <repo> -> custom_nodes -> <ComfyUI root>
$RepoRoot  = (Get-Item $PSScriptRoot).Parent.FullName
$ComfyRoot = (Get-Item $RepoRoot).Parent.Parent.FullName
$Root = if ($env:OTR_INDEXTTS2_ROOT) { $env:OTR_INDEXTTS2_ROOT } else { Join-Path $ComfyRoot "index-tts" }
# Tested pin (2026-07-10, this box: py 3.10.20 / torch 2.8.0+cu128). A newer
# commit may work -- re-derive the pin deliberately, never by accident.
$Pin = "830f6f8f94a51fea23ab1d639027a86200075a4e"

Write-Host "IndexTTS2 Path-B install -> $Root"

# 1) uv
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found -- bootstrapping via the astral.sh installer"
    powershell -ExecutionPolicy ByPass -NoProfile -Command "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv bootstrap failed -- install uv manually (https://docs.astral.sh/uv/) and re-run"
    }
}

# 2) clone + pin
if (-not (Test-Path (Join-Path $Root ".git"))) {
    git clone https://github.com/index-tts/index-tts.git $Root
}
git -C $Root fetch --all --quiet
git -C $Root checkout $Pin

# 3) isolated venv + locked deps. uv.lock is the whole dependency story --
#    do NOT hand-pin torch/numpy against the main venv.
Push-Location $Root
try { uv sync } finally { Pop-Location }
$VPy = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $VPy)) {
    # Mac/Linux layout fallback (recipe reuse): bin/python
    $VPy = Join-Path $Root ".venv/bin/python"
}
if (-not (Test-Path $VPy)) { throw "uv sync did not produce a venv python under $Root\.venv" }

# 4) weights (fail-loud size/manifest validation inside the script)
& $VPy (Join-Path $RepoRoot "scripts\_otr_idx_download_weights.py")
if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 weights download/validation failed (exit $LASTEXITCODE)" }

# 5) readiness smoke: boot the worker once (loads the model -- slow first
#    time); it must print {"ready": true}, then the stop request ends it.
$Worker = Join-Path $RepoRoot "scripts\_otr_indextts2_worker.py"
$Ckpt   = Join-Path $Root "checkpoints"
$ready = '{"stop": true}' | & $VPy $Worker --model-dir $Ckpt | Select-Object -First 1
Write-Host "worker readiness: $ready"
if ($ready -notmatch '"ready":\s*true') {
    throw "IndexTTS2 worker readiness smoke FAILED: $ready (see _otr_indextts2_worker.err next to the repo root)"
}

Write-Host ""
Write-Host "Done. indextts2 (the default char voice) is installed at $Root."
Write-Host "Optional env overrides: OTR_INDEXTTS2_ROOT, OTR_INDEXTTS2_VENV, OTR_INDEXTTS2_DIR, OTR_INDEXTTS2_WORKER, OTR_INDEXTTS2_FP16=1."
Write-Host "Restart ComfyUI before rendering."
