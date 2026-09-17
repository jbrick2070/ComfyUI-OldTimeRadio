$ErrorActionPreference = "Stop"
$roots = @(
    "C:\ComfyUI-Models",
    "C:\Users\jeffr\Documents\ComfyUI\models",
    "C:\Users\jeffr\ComfyUI-Installs\ComfyUI (1)\ComfyUI\models"
)
$names = @(
    "flux-2-klein-4b-Q4_K_M.gguf",
    "flux2-vae.safetensors",
    "qwen_3_4b.safetensors",
    "qwen_3_4b_fp4_flux2.safetensors",
    "flux1-dev-fp8.safetensors"
)
foreach ($root in $roots) {
    Write-Output ("ROOT " + $root + " exists=" + (Test-Path $root))
}
foreach ($name in $names) {
    $hits = @()
    foreach ($root in $roots) {
        if (Test-Path $root) {
            $hits += Get-ChildItem -Path $root -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName
        }
    }
    if ($hits.Count -eq 0) { Write-Output ("MISSING " + $name) }
    else { foreach ($h in $hits) { Write-Output ("FOUND " + $h) } }
}
