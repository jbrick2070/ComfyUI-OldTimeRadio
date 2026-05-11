# download_ltx_2_3.ps1 -- canonical-storage download of the Kijai LTX 2.3
# distilled FP8 model.
#
# Pulls the file via huggingface-cli (uses your existing venv's hf cache,
# respects HF_HOME=C:\ComfyUI-Models\huggingface), then creates a symlink
# at C:\ComfyUI-Models\diffusion_models\<file>.safetensors pointing into
# the cache blob. Single source of truth on disk; ComfyUI's diffusion_models
# loader sees the symlink and resolves the file transparently.
#
# Usage (from any PowerShell prompt):
#   PS> C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\scripts\download_ltx_2_3.ps1
#
# Requires: ComfyUI venv already exists at C:\Users\jeffr\Documents\ComfyUI\.venv
# Requires: Windows Developer Mode ON (for non-admin symlink) OR run as admin.
# Disk: ~22 GB free needed.

$ErrorActionPreference = 'Stop'

$REPO     = 'Kijai/LTX2.3_comfy'
$FILE     = 'diffusion_models/ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors'
$VENV_PY  = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$VENV_HF  = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\huggingface-cli.exe'
$HF_HOME  = 'C:\ComfyUI-Models\huggingface'
$DM_DIR   = 'C:\ComfyUI-Models\diffusion_models'
$DM_NAME  = 'ltx-2.3-22b-distilled-1.1_transformer_only_fp8_scaled.safetensors'
$DM_LINK  = Join-Path $DM_DIR $DM_NAME

if (-not (Test-Path $VENV_HF)) {
    Write-Error "huggingface-cli not found at $VENV_HF -- is the venv intact?"
    exit 2
}

$env:HF_HOME = $HF_HOME
Write-Host "[ltx-2.3] HF_HOME = $($env:HF_HOME)"
Write-Host "[ltx-2.3] downloading $REPO :: $FILE"
Write-Host "[ltx-2.3] expect ~22 GB; will go to HF cache under $HF_HOME"

& $VENV_HF download $REPO $FILE
if ($LASTEXITCODE -ne 0) {
    Write-Error "huggingface-cli download failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

# Resolve the cached blob path via huggingface_hub's own resolver. We use
# try_to_load_from_cache so we get the exact path (which includes the
# snapshot hash) without re-downloading or hitting the network.
$resolveScript = @"
from huggingface_hub import try_to_load_from_cache
p = try_to_load_from_cache(repo_id='$REPO', filename='$FILE')
print(p if p else '')
"@

$cachedPath = & $VENV_PY -c $resolveScript
$cachedPath = $cachedPath.Trim()

if (-not $cachedPath -or -not (Test-Path $cachedPath)) {
    Write-Error "could not resolve cached path; download may have failed"
    exit 3
}

Write-Host "[ltx-2.3] cached at: $cachedPath"

# Create the symlink under diffusion_models/ if it doesn't exist
if (-not (Test-Path $DM_DIR)) {
    New-Item -ItemType Directory -Path $DM_DIR | Out-Null
}

if (Test-Path $DM_LINK) {
    Write-Host "[ltx-2.3] symlink already exists at: $DM_LINK"
    Write-Host "[ltx-2.3] target points at: $((Get-Item $DM_LINK).Target)"
} else {
    try {
        New-Item -ItemType SymbolicLink -Path $DM_LINK -Target $cachedPath | Out-Null
        Write-Host "[ltx-2.3] symlinked: $DM_LINK -> $cachedPath"
    } catch {
        Write-Warning "Symlink failed (likely needs Developer Mode or admin): $_"
        Write-Warning "Falling back to junction (works without admin on NTFS)..."
        # Junction: works for directories on NTFS; for files we need a HARD LINK
        # which doesn't require admin / dev mode and is transparent to readers.
        try {
            New-Item -ItemType HardLink -Path $DM_LINK -Target $cachedPath | Out-Null
            Write-Host "[ltx-2.3] hardlinked: $DM_LINK -> $cachedPath"
        } catch {
            Write-Error "Both symlink and hardlink failed: $_"
            Write-Error "Manual fix: copy the file:  Copy-Item '$cachedPath' '$DM_LINK'"
            exit 4
        }
    }
}

Write-Host ""
Write-Host "[ltx-2.3] DONE. Reload ComfyUI and the new model should appear in"
Write-Host "[ltx-2.3] the diffusion-model dropdown of the LTX 2.3 workflow."
