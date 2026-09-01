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
#   3) uv sync --frozen --python 3.10 (creates the locked isolated environment)
#      project editable so `import indextts` resolves in the worker)
#   4) download + validate weights via scripts\_otr_idx_download_weights.py
#   5) readiness smoke via scripts\_otr_indextts2_worker.py
#
# Re-run safe: every step is idempotent.
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 does not turn a native command's non-zero exit into a
# terminating error. Every git/uv step therefore goes through this owner; a
# stale venv must never let a failed checkout or sync continue to "Done".
function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory = $true)][string]$Label,
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$ArgumentList
    )
    $Output = & $FilePath @ArgumentList
    $Code = $LASTEXITCODE
    if ($Code -ne 0) {
        throw "$Label failed (exit $Code)"
    }
    return $Output
}

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
    $null = Invoke-NativeChecked "uv bootstrap" "powershell" @(
        "-ExecutionPolicy", "ByPass", "-NoProfile", "-Command",
        "irm https://astral.sh/uv/install.ps1 | iex"
    )
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
        throw "uv bootstrap failed -- install uv manually (https://docs.astral.sh/uv/) and re-run"
    }
}

# 2) clone + pin
if (-not (Test-Path (Join-Path $Root ".git"))) {
    $null = Invoke-NativeChecked "IndexTTS2 clone" "git" @(
        "clone", "https://github.com/index-tts/index-tts.git", $Root
    )
}
$null = Invoke-NativeChecked "IndexTTS2 fetch" "git" @(
    "-C", $Root, "fetch", "--all", "--quiet"
)
$null = Invoke-NativeChecked "IndexTTS2 checkout" "git" @(
    "-C", $Root, "checkout", "--detach", $Pin
)
$Head = [string](Invoke-NativeChecked "IndexTTS2 HEAD probe" "git" @(
    "-C", $Root, "rev-parse", "HEAD"
) | Select-Object -Last 1)
if ($Head.Trim() -cne $Pin) {
    throw "IndexTTS2 HEAD $($Head.Trim()) != pinned $Pin"
}
$Drift = @(Invoke-NativeChecked "IndexTTS2 source drift probe" "git" @(
    "-C", $Root, "status", "--porcelain=v1", "--untracked-files=all", "--", ".",
    ":(exclude).venv", ":(exclude).venv/**",
    ":(exclude).uv-python", ":(exclude).uv-python/**",
    ":(exclude)checkpoints", ":(exclude)checkpoints/**"
))
if ($Drift.Count -ne 0) {
    throw "IndexTTS2 source drift outside .venv/.uv-python/checkpoints: $($Drift -join '; ')"
}

# 3) isolated venv + locked deps. uv.lock is the whole dependency story --
#    do NOT hand-pin torch/numpy against the main venv. Keep uv's managed
#    interpreter beside the venv so a RunPod recreate cannot leave a surviving
#    network-volume venv pointing into erased container-home storage.
$PriorUvPythonDir = $env:UV_PYTHON_INSTALL_DIR
$PriorUvPythonPreference = $env:UV_PYTHON_PREFERENCE
try {
    if (-not $env:UV_PYTHON_INSTALL_DIR) {
        $env:UV_PYTHON_INSTALL_DIR = Join-Path $Root ".uv-python"
    }
    $env:UV_PYTHON_PREFERENCE = "only-managed"
    $null = Invoke-NativeChecked "IndexTTS2 managed Python install" "uv" @(
        "python", "install", "3.10"
    )
    $null = Invoke-NativeChecked "IndexTTS2 locked uv sync" "uv" @(
        "sync", "--frozen", "--python", "3.10", "--project", $Root
    )
} finally {
    $env:UV_PYTHON_INSTALL_DIR = $PriorUvPythonDir
    $env:UV_PYTHON_PREFERENCE = $PriorUvPythonPreference
}
$VPy = if ($env:OTR_INDEXTTS2_VENV) { $env:OTR_INDEXTTS2_VENV } else { Join-Path $Root ".venv\Scripts\python.exe" }
if (-not $env:OTR_INDEXTTS2_VENV -and -not (Test-Path $VPy)) {
    # Mac/Linux layout fallback (recipe reuse): bin/python
    $VPy = Join-Path $Root ".venv/bin/python"
}
if (-not (Test-Path $VPy)) { throw "uv sync did not produce a venv python under $Root\.venv" }

# 4) weights (fail-loud size/manifest validation inside the script). Set the
# exact target for this child so ROOT/VENV overrides cannot split source and
# checkpoints across two trees.
$Ckpt = if ($env:OTR_INDEXTTS2_DIR) { $env:OTR_INDEXTTS2_DIR } else { Join-Path $Root "checkpoints" }
if ((Split-Path -Leaf $Ckpt) -cne "checkpoints") {
    throw "OTR_INDEXTTS2_DIR must end in literal lower-case 'checkpoints'"
}
$PriorIndexDir = $env:OTR_INDEXTTS2_DIR
try {
    $env:OTR_INDEXTTS2_DIR = $Ckpt
    & $VPy (Join-Path $RepoRoot "scripts\_otr_idx_download_weights.py")
    if ($LASTEXITCODE -ne 0) { throw "IndexTTS2 weights download/validation failed (exit $LASTEXITCODE)" }
} finally {
    $env:OTR_INDEXTTS2_DIR = $PriorIndexDir
}

# 5) readiness smoke: boot the worker once (loads the model -- slow first
#    time); it must print {"ready": true}, then the stop request ends it.
$Worker = if ($env:OTR_INDEXTTS2_WORKER) { $env:OTR_INDEXTTS2_WORKER } else { Join-Path $RepoRoot "scripts\_otr_indextts2_worker.py" }
$WorkerArgs = @($Worker, "--model-dir", $Ckpt)
if ($env:OTR_INDEXTTS2_FP16 -eq "1") { $WorkerArgs += "--fp16" }
$PriorHfOffline = $env:HF_HUB_OFFLINE
$PriorTransformersOffline = $env:TRANSFORMERS_OFFLINE
try {
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    Push-Location (Split-Path -Parent $Ckpt)
    try {
        $ready = '{"stop": true}' | & $VPy @WorkerArgs | Select-Object -First 1
        $WorkerExit = $LASTEXITCODE
    } finally {
        Pop-Location
    }
} finally {
    $env:HF_HUB_OFFLINE = $PriorHfOffline
    $env:TRANSFORMERS_OFFLINE = $PriorTransformersOffline
}
Write-Host "worker readiness: $ready"
if ($WorkerExit -ne 0) {
    throw "IndexTTS2 worker readiness process failed (exit $WorkerExit)"
}
if ($ready -notmatch '"ready":\s*true') {
    throw "IndexTTS2 worker readiness smoke FAILED: $ready (see _otr_indextts2_worker.err next to the repo root)"
}

Write-Host ""
Write-Host "Done. indextts2 (the default char voice) is installed at $Root."
Write-Host "Optional env overrides: OTR_INDEXTTS2_ROOT, OTR_INDEXTTS2_VENV, OTR_INDEXTTS2_DIR, OTR_INDEXTTS2_WORKER, OTR_INDEXTTS2_FP16=1."
Write-Host "Restart ComfyUI before rendering."
