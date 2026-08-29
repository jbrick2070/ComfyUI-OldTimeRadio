# OTR Dia Path-B install -- isolated venv (Nari Labs Dia, Apache-2.0 ->
# COMMERCIAL-CLEAN). Blackwell (RTX 5080, sm_120) needs a CUDA cu128 torch
# build, which would conflict with the main venv -> isolate it.
# After it finishes: RESTART ComfyUI (no enable flag -- installing IS the opt-in).
$ErrorActionPreference = "Stop"
$Root = "C:\Users\jeffr\Documents\ComfyUI\dia"
$Venv = Join-Path $Root ".venv"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1) Create the isolated venv (Python 3.10/3.11).
if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
$VPy = Join-Path $Venv "Scripts\python.exe"
& $VPy -m pip install --upgrade pip

# 2) Dia from source (Apache-2.0). Its package currently pins older torch wheels,
#    so install it BEFORE the Blackwell torch repair below.
& $VPy -m pip install "git+https://github.com/nari-labs/dia.git"

# 3) Blackwell torch LAST. Dia's dependency install can otherwise overwrite this
#    with CPU torch and make CUDA silently unavailable.
& $VPy -m pip install --pre --upgrade --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 4) Runtime audio helper. The worker loads WAV refs with soundfile and passes
#    DAC codes to Dia, avoiding TorchCodec's Windows shared-FFmpeg dependency.
& $VPy -m pip install --upgrade soundfile

# 5) Smoke: import + cuda (NOT a render -- that downloads ~weights on first use).
& $VPy -c "import torch; from dia.model import Dia; print('dia OK -- torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host ("  (optional) setx OTR_DIA_VENV " + '"' + $VPy + '"')
Write-Host "  Restart ComfyUI. On the canonical workflow set node 80 voice_bank=default"
Write-Host "  and node 81 engine=dia, then queue a small cast."
Write-Host "  (optional quality upgrade) add config\dia_ref_transcripts.json keyed by"
Write-Host "  reference WAV basename to enable transcript-conditioned cloning."
