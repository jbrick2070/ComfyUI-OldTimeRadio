# audit_video_stack_weights.ps1
# -----------------------------------------------------------------------------
# Reports presence of each video stack weight target, matching the defaults
# in otr_v2/hyworld/backends/*.py. Read-only — never downloads, never edits.
# -----------------------------------------------------------------------------

$ModelsRoot = "C:\Users\jeffr\Documents\ComfyUI\models"

$Checks = @(
    @{ name = "FLUX.1-dev (base)";              path = "$ModelsRoot\diffusers\FLUX.1-dev";                                     kind = "dir_has_model_index" },
    @{ name = "PuLID-FLUX";                     path = "$ModelsRoot\pulid\pulid_flux.safetensors";                             kind = "file" },
    @{ name = "ControlNet Union Pro 2.0";       path = "$ModelsRoot\controlnet\FLUX.1-dev-ControlNet-Union-Pro-2.0";           kind = "dir_has_safetensors" },
    @{ name = "ControlNet Depth (fallback)";    path = "$ModelsRoot\controlnet\FLUX.1-dev-ControlNet-Depth";                   kind = "dir_has_safetensors" },
    @{ name = "Wan2.1 I2V 1.3B";                path = "$ModelsRoot\diffusers\Wan2.1-I2V-1.3B";                                kind = "dir_has_model_index" },
    @{ name = "Florence-2 Large";               path = "$ModelsRoot\florence2\Florence-2-large";                               kind = "dir_has_config" },
    @{ name = "SDXL Inpaint 1.0";               path = "$ModelsRoot\diffusers\stable-diffusion-xl-1.0-inpainting-0.1";         kind = "dir_has_model_index" },
    @{ name = "LTX-2.3 22B FP8 (baseline)";     path = "$ModelsRoot\checkpoints\ltx-video-2b-v0.9.safetensors";                 kind = "file_any_ltx" }
)

function Test-Target {
    param($entry)
    $p = $entry.path
    switch ($entry.kind) {
        "file"                  { return (Test-Path $p -PathType Leaf) }
        "dir_has_model_index"   {
            if (-not (Test-Path $p -PathType Container)) { return $false }
            return (Test-Path (Join-Path $p "model_index.json"))
        }
        "dir_has_safetensors"   {
            if (-not (Test-Path $p -PathType Container)) { return $false }
            return (@(Get-ChildItem -Path $p -Filter *.safetensors -ErrorAction SilentlyContinue).Count -gt 0)
        }
        "dir_has_config"        {
            if (-not (Test-Path $p -PathType Container)) { return $false }
            return (Test-Path (Join-Path $p "config.json"))
        }
        "file_any_ltx"          {
            # LTX filename varies (2b, 22b, dev, distilled); just check the dir
            $dir = Split-Path $p -Parent
            if (-not (Test-Path $dir -PathType Container)) { return $false }
            return (@(Get-ChildItem -Path $dir -Filter *ltx*.safetensors -ErrorAction SilentlyContinue).Count -gt 0)
        }
    }
    return $false
}

$PresentCount = 0
Write-Host "=== Video stack weights audit ==="
foreach ($c in $Checks) {
    $ok = Test-Target -entry $c
    if ($ok) { $PresentCount++ }
    $mark = if ($ok) { "PRESENT" } else { "MISSING" }
    Write-Host ("{0,-8}  {1,-32}  {2}" -f $mark, $c.name, $c.path)
}
Write-Host "---"
Write-Host ("Present: {0}/{1}" -f $PresentCount, $Checks.Count)
if ($PresentCount -ne $Checks.Count) { exit 1 } else { exit 0 }
