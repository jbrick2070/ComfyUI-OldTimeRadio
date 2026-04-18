# download_video_stack_weights.ps1
# -----------------------------------------------------------------------------
# Idempotent HuggingFace snapshot download for the SIGNAL LOST video stack.
# Run while away — resumable, skips already-present files.
#
# Requires: Python 3.12 venv at C:\Users\jeffr\Documents\ComfyUI\.venv with
# `huggingface_hub` installed (ships with diffusers / transformers).
# HF_TOKEN must be set in environment for FLUX.1-dev (gated repo).
#
# Canonical sources + target paths match:
#   docs/2026-04-17-weights-download-plan.md
#
# Logs: logs\weights_download.log (gitignored)
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"

$RepoRoot = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$ModelsRoot = "C:\Users\jeffr\Documents\ComfyUI\models"
$Python = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$LogDir = Join-Path $RepoRoot "logs"
$LogFile = Join-Path $LogDir "weights_download.log"

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }

function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== Video stack weights download start ==="
Log "Python: $Python"
Log "Models root: $ModelsRoot"

if (-not (Test-Path $Python)) {
    Log "FATAL: venv python not found at $Python"
    exit 2
}

# -----------------------------------------------------------------------------
# Manifest: HF repo -> local target dir (or file)
# -----------------------------------------------------------------------------
$Manifest = @(
    @{ repo = "black-forest-labs/FLUX.1-dev";
       target = "$ModelsRoot\diffusers\FLUX.1-dev";
       kind = "snapshot"; gated = $true; approx_gb = 23.0 },
    @{ repo = "guozinan/PuLID";
       target = "$ModelsRoot\pulid";
       kind = "file"; file = "pulid_flux.safetensors"; approx_gb = 1.2 },
    @{ repo = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0";
       target = "$ModelsRoot\controlnet\FLUX.1-dev-ControlNet-Union-Pro-2.0";
       kind = "snapshot"; approx_gb = 6.6 },
    @{ repo = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth";
       target = "$ModelsRoot\controlnet\FLUX.1-dev-ControlNet-Depth";
       kind = "snapshot"; approx_gb = 1.4 },
    @{ repo = "depth-anything/Depth-Anything-V2-Large-hf";
       target = "HF_CACHE";
       kind = "snapshot_to_cache"; approx_gb = 1.2 },
    @{ repo = "Wan-AI/Wan2.1-I2V-1.3B";
       target = "$ModelsRoot\diffusers\Wan2.1-I2V-1.3B";
       kind = "snapshot"; approx_gb = 5.5 },
    @{ repo = "microsoft/Florence-2-large";
       target = "$ModelsRoot\florence2\Florence-2-large";
       kind = "snapshot"; approx_gb = 1.5 },
    @{ repo = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1";
       target = "$ModelsRoot\diffusers\stable-diffusion-xl-1.0-inpainting-0.1";
       kind = "snapshot"; approx_gb = 6.9 }
)

$TotalGb = ($Manifest | Measure-Object -Property approx_gb -Sum).Sum
Log "Total download estimate: $TotalGb GB (skip-existing enabled)"

# -----------------------------------------------------------------------------
# Downloader invoked as a single python child for each entry.
# Uses huggingface_hub snapshot_download / hf_hub_download; cache is the
# default HF cache so re-running is idempotent.
# -----------------------------------------------------------------------------
$PyDownloader = @'
import os, sys, json, traceback
from huggingface_hub import snapshot_download, hf_hub_download
spec = json.loads(sys.argv[1])
repo = spec["repo"]
target = spec["target"]
kind = spec["kind"]
token = os.environ.get("HF_TOKEN") or None

try:
    if kind == "snapshot":
        os.makedirs(target, exist_ok=True)
        out = snapshot_download(
            repo_id=repo,
            local_dir=target,
            local_dir_use_symlinks=False,
            token=token,
            allow_patterns=None,
            ignore_patterns=["*.gguf", "*.onnx", "*_fp4.safetensors", "*.msgpack"],
        )
        print(f"OK snapshot {repo} -> {out}")
    elif kind == "snapshot_to_cache":
        out = snapshot_download(repo_id=repo, token=token)
        print(f"OK cache {repo} -> {out}")
    elif kind == "file":
        os.makedirs(target, exist_ok=True)
        out = hf_hub_download(
            repo_id=repo,
            filename=spec["file"],
            local_dir=target,
            local_dir_use_symlinks=False,
            token=token,
        )
        print(f"OK file {repo}::{spec['file']} -> {out}")
    else:
        print(f"SKIP unknown kind {kind}")
        sys.exit(3)
except Exception as exc:
    print(f"FAIL {repo}: {type(exc).__name__}: {exc}")
    traceback.print_exc()
    sys.exit(1)
'@

$TmpScript = Join-Path $env:TEMP "otr_hf_download.py"
Set-Content -Path $TmpScript -Value $PyDownloader -Encoding UTF8

$Failed = @()
$Ok = @()

foreach ($entry in $Manifest) {
    $repo = $entry.repo
    $target = $entry.target
    $gated = $false
    if ($entry.ContainsKey("gated")) { $gated = $entry.gated }

    if ($gated -and -not $env:HF_TOKEN) {
        Log "SKIP $repo — gated, HF_TOKEN not set in this session"
        $Failed += $repo
        continue
    }

    Log "--> $repo (~$($entry.approx_gb) GB) -> $target"
    $specJson = ($entry | ConvertTo-Json -Compress)
    & $Python $TmpScript $specJson 2>&1 | ForEach-Object { Log $_ }
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Log "OK   $repo"
        $Ok += $repo
    } else {
        Log "FAIL $repo exit=$rc"
        $Failed += $repo
    }
}

Remove-Item $TmpScript -Force -ErrorAction SilentlyContinue

Log "=== Download complete ==="
Log "OK:     $($Ok.Count) / $($Manifest.Count)"
Log "FAIL:   $($Failed.Count) / $($Manifest.Count)"
if ($Failed.Count -gt 0) {
    Log "Failed repos:"
    $Failed | ForEach-Object { Log "  - $_" }
    exit 1
}
Log "All weights present. Next: scripts\audit_video_stack_weights.ps1"
exit 0
