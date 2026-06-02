# ===========================================================================
# OTR determinism launcher (C-1 / C-7) -- THE entrypoint for OTR runs.
# Exports the determinism env BEFORE python/torch import, then starts ComfyUI
# from the project venv. If your ComfyUI entry script is not main.py, adjust
# the final line; the env block above it must stay first.
# ===========================================================================
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
$env:PYTHONHASHSEED = '0'
$env:NVIDIA_TF32_OVERRIDE = '0'
$env:TOKENIZERS_PARALLELISM = 'false'

$ComfyRoot = 'C:\Users\jeffr\Documents\ComfyUI'
$VenvPy = Join-Path $ComfyRoot '.venv\Scripts\python.exe'

Set-Location $ComfyRoot
& $VenvPy main.py @args
