# download_video_stack_weights.ps1
# -----------------------------------------------------------------------------
# Idempotent HuggingFace snapshot download for the SIGNAL LOST video stack.
# Run while away - resumable, skips already-present files.
#
# Requires:
#   - Python 3.12 venv at C:\Users\jeffr\Documents\ComfyUI\.venv with
#     huggingface_hub installed (ships with diffusers / transformers).
#   - HF_TOKEN env var for FLUX.1-dev (gated).  Accept the license first at
#     https://huggingface.co/black-forest-labs/FLUX.1-dev .
#
# Canonical sources + target paths match docs/2026-04-17-weights-download-plan.md
# Logs: logs\weights_download.log (gitignored)
# -----------------------------------------------------------------------------

$ErrorActionPreference = "Continue"
$ProgressPreference    = "SilentlyContinue"

$RepoRoot   = "C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio"
$ModelsRoot = "C:\Users\jeffr\Documents\ComfyUI\models"
$Python     = "C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe"
$Driver     = Join-Path $RepoRoot "scripts\hf_download_driver.py"
$LogDir     = Join-Path $RepoRoot "logs"
$LogFile    = Join-Path $LogDir "weights_download.log"

if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

function Log {
    param([string]$Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $line = "[$timestamp] $Message"
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

Log "=== Video stack weights download start ==="
Log "Python: $Python"
Log "Driver: $Driver"
Log "Models root: $ModelsRoot"

if (-not (Test-Path $Python)) {
    Log "FATAL: venv python not found at $Python"
    exit 2
}
if (-not (Test-Path $Driver)) {
    Log "FATAL: hf_download_driver.py not found at $Driver"
    exit 2
}

# -----------------------------------------------------------------------------
# HF_TOKEN: pull from User-scope env var if not already in the in-process
# environment.  Older terminal sessions opened before setx won't have the
# in-process copy, so read HKCU\Environment directly and populate $env:.
# Never logs the token value - only presence + length.
# -----------------------------------------------------------------------------
if (-not $env:HF_TOKEN) {
    $userTok = [Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
    if ($userTok) {
        $env:HF_TOKEN = $userTok
        Log ("HF_TOKEN pulled from User-scope env (length=" + $userTok.Length + ")")
    } else {
        Log "HF_TOKEN not set at User or Process scope - gated repos will be skipped"
    }
} else {
    Log ("HF_TOKEN already in process env (length=" + $env:HF_TOKEN.Length + ")")
}

# -----------------------------------------------------------------------------
# Manifest: HF repo -> local target dir (or file).  Kept as ordered list so
# the big gated FLUX repo goes first (fail fast on missing HF_TOKEN).
# -----------------------------------------------------------------------------
$Manifest = @(
    @{ repo   = "black-forest-labs/FLUX.1-dev"
       target = (Join-Path $ModelsRoot "diffusers\FLUX.1-dev")
       kind   = "snapshot"
       gated  = $true
       approx_gb = 23.0 }

    @{ repo   = "guozinan/PuLID"
       target = (Join-Path $ModelsRoot "pulid")
       kind   = "snapshot"
       # Upstream has re-versioned the weights file (pulid_flux_v0.9.1.safetensors,
       # pulid_flux_v0.9.0.safetensors, etc.).  A glob keeps us forward-compatible
       # without pinning a specific filename.  Backends already discover by prefix.
       allow_patterns = @("pulid_flux*.safetensors", "pulid_flux*.bin")
       approx_gb = 1.2 }

    @{ repo   = "Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0"
       target = (Join-Path $ModelsRoot "controlnet\FLUX.1-dev-ControlNet-Union-Pro-2.0")
       kind   = "snapshot"
       approx_gb = 6.6 }

    @{ repo   = "Shakker-Labs/FLUX.1-dev-ControlNet-Depth"
       target = (Join-Path $ModelsRoot "controlnet\FLUX.1-dev-ControlNet-Depth")
       kind   = "snapshot"
       approx_gb = 1.4 }

    @{ repo   = "depth-anything/Depth-Anything-V2-Large-hf"
       target = "HF_CACHE"
       kind   = "snapshot_to_cache"
       approx_gb = 1.2 }

    # Wan-AI never published a 1.3B I2V variant - only the T2V-1.3B
    # (Diffusers-formatted) repo exists at that weight class.  The
    # wan21_loop backend gracefully falls back to T2V when the I2V
    # pipeline class is unavailable, so we pull the T2V weights into
    # the historical I2V target path and the backend finds them.
    @{ repo   = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
       target = (Join-Path $ModelsRoot "diffusers\Wan2.1-I2V-1.3B")
       kind   = "snapshot"
       approx_gb = 5.5 }

    @{ repo   = "microsoft/Florence-2-large"
       target = (Join-Path $ModelsRoot "florence2\Florence-2-large")
       kind   = "snapshot"
       approx_gb = 1.5 }

    @{ repo   = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"
       target = (Join-Path $ModelsRoot "diffusers\stable-diffusion-xl-1.0-inpainting-0.1")
       kind   = "snapshot"
       approx_gb = 6.9 }
)

$TotalGb = 0.0
foreach ($entry in $Manifest) { $TotalGb += [double]$entry.approx_gb }
Log "Total download estimate: $TotalGb GB (skip-existing enabled via HF cache)"

$Failed  = @()
$Ok      = @()
$SpecDir = Join-Path $env:TEMP "otr_hf_specs"
if (-not (Test-Path $SpecDir)) { New-Item -ItemType Directory -Path $SpecDir | Out-Null }

foreach ($entry in $Manifest) {
    $repo   = $entry.repo
    $target = $entry.target
    $gated  = $false
    if ($entry.ContainsKey("gated")) { $gated = $entry.gated }

    if ($gated -and -not $env:HF_TOKEN) {
        Log ("SKIP " + $repo + " -- gated, HF_TOKEN not set in this session")
        $Failed += $repo
        continue
    }

    Log ("--> " + $repo + " (~" + $entry.approx_gb + " GB) -> " + $target)

    # Write each spec to a temp JSON file so cmd.exe quoting can't mangle it
    # on the argv boundary.  Python driver reads the path, not the JSON body.
    $safeName = ($repo -replace "[^A-Za-z0-9]", "_") + ".json"
    $specFile = Join-Path $SpecDir $safeName
    $jsonBody = ($entry | ConvertTo-Json -Compress)
    # UTF-8 without BOM (CLAUDE.md rule: always no BOM)
    [System.IO.File]::WriteAllText($specFile, $jsonBody, (New-Object System.Text.UTF8Encoding($false)))

    & $Python $Driver $specFile 2>&1 | ForEach-Object { Log $_ }
    $rc = $LASTEXITCODE
    if ($rc -eq 0) {
        Log ("OK   " + $repo)
        $Ok += $repo

        # PuLID canonicalization: upstream re-versioned the weights file
        # (pulid_flux_v0.9.1.safetensors, pulid_flux_v0.9.0.safetensors, ...),
        # but the backend default expects pulid_flux.safetensors.  If the
        # canonical name is missing, hard-link it to the newest versioned
        # match so no production code changes are required.  Hard link is
        # instant on NTFS same-volume, uses zero disk.
        if ($repo -eq "guozinan/PuLID") {
            $canonical = Join-Path $target "pulid_flux.safetensors"
            if (-not (Test-Path $canonical)) {
                $found = @(Get-ChildItem -Path $target -Filter "pulid_flux*.safetensors" -ErrorAction SilentlyContinue | Sort-Object Name)
                if ($found.Count -gt 0) {
                    $src = $found[-1].FullName
                    try {
                        New-Item -ItemType HardLink -Path $canonical -Target $src -ErrorAction Stop | Out-Null
                        Log ("     hardlink pulid_flux.safetensors -> " + $found[-1].Name)
                    } catch {
                        Copy-Item -Path $src -Destination $canonical -Force
                        Log ("     copy pulid_flux.safetensors <- " + $found[-1].Name + " (hardlink failed)")
                    }
                } else {
                    Log ("     WARN: no pulid_flux*.safetensors found in " + $target)
                }
            }
        }
    } else {
        Log ("FAIL " + $repo + " exit=" + $rc)
        $Failed += $repo
    }
}

Remove-Item -Path $SpecDir -Recurse -Force -ErrorAction SilentlyContinue

Log "=== Download complete ==="
Log ("OK:     " + $Ok.Count + " / " + $Manifest.Count)
Log ("FAIL:   " + $Failed.Count + " / " + $Manifest.Count)
if ($Failed.Count -gt 0) {
    Log "Failed repos:"
    $Failed | ForEach-Object { Log ("  - " + $_) }
    exit 1
}
Log "All weights present. Next: scripts\audit_video_stack_weights.ps1"
exit 0
