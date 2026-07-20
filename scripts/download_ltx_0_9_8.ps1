# download_ltx_0_9_8.ps1 -- fetch + integrity-verify the LTX 0.9.8 distilled 2B
# checkpoint for the ltx_8gb video engine (video-tiers build, 2026-07-20).
#
# Self-driving: builds a pinned-revision + size/SHA-verified spec and hands it to
# scripts/hf_download_driver.py (the extended "file" kind). The SHA-256 is the real
# integrity guarantee; the pinned commit is defense-in-depth against a repo re-tag.
# The 0.9.x all-in-one checkpoint embeds the VAE; the T5 text encoder is the shared
# t5xxl_fp16.safetensors already on disk (NOT fetched here). NO fp8 fork.
#
# UTF-8, no BOM, ASCII-only. cd-first; ASCII quotes; copy-paste ready.

$ErrorActionPreference = 'Stop'

$RepoRoot = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$VenvPy   = 'C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe'
$Driver   = Join-Path $RepoRoot 'scripts\hf_download_driver.py'
$Dest     = 'C:\ComfyUI-Models\checkpoints'
Set-Location $RepoRoot

$spec = [ordered]@{
    repo            = 'Lightricks/LTX-Video'
    kind            = 'file'
    file            = 'ltxv-2b-0.9.8-distilled.safetensors'
    target          = $Dest
    # Pinned to the repo commit that carries this weight (HF api sha 2025-07-16).
    revision        = '8984fa25007f376c1a299016d0957a37a2f797bb'
    expected_size   = 6340744492
    expected_sha256 = '76aa8c4786af752fa6f951947129d5290c3c6c0b2fadcadea6b5e114ae2cad8f'
}

$SpecPath = Join-Path $env:TEMP ('otr_ltx098_spec_{0}.json' -f ([guid]::NewGuid().ToString('N')))
$json = $spec | ConvertTo-Json -Depth 4
# Write UTF-8 no BOM.
[System.IO.File]::WriteAllText($SpecPath, $json, (New-Object System.Text.UTF8Encoding($false)))

Write-Host ('[download_ltx_0_9_8] fetching {0}::{1} -> {2}' -f $spec.repo, $spec.file, $Dest)
$env:PYTHONUTF8 = '1'
& $VenvPy $Driver $SpecPath
$code = $LASTEXITCODE
Remove-Item $SpecPath -ErrorAction SilentlyContinue

if ($code -ne 0) {
    Write-Error ('[download_ltx_0_9_8] driver failed rc={0} (integrity or fetch error)' -f $code)
    exit $code
}
$Final = Join-Path $Dest $spec.file
if (-not (Test-Path $Final)) {
    Write-Error ('[download_ltx_0_9_8] expected asset missing after download: {0}' -f $Final)
    exit 1
}
Write-Host ('[download_ltx_0_9_8] OK verified {0} ({1} bytes)' -f $Final, (Get-Item $Final).Length)
exit 0
