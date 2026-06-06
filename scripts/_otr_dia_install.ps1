# OTR Dia Path-B install -- isolated venv (Nari Labs Dia, Apache-2.0 ->
# COMMERCIAL-CLEAN). Blackwell (RTX 5080, sm_120) needs torch 2.8 nightly cu128
# (Nari issue #26), which would conflict with the main venv -> isolate it.
# After it finishes: setx OTR_ENABLE_DIA 1, then RESTART ComfyUI.
$ErrorActionPreference = "Stop"
$Root = "C:\Users\jeffr\Documents\ComfyUI\dia"
$Venv = Join-Path $Root ".venv"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1) Create the isolated venv (Python 3.10/3.11).
if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
$VPy = Join-Path $Venv "Scripts\python.exe"
& $VPy -m pip install --upgrade pip

# 2) Blackwell torch FIRST (torch 2.8 nightly cu128). If a stable cu128 torch
#    >= 2.8 exists when you run this, prefer it (drop --pre + the nightly index).
& $VPy -m pip install --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# 3) Dia from source (Apache-2.0). First model load downloads Dia-1.6B-0626 + the
#    Descript Audio Codec.
& $VPy -m pip install "git+https://github.com/nari-labs/dia.git"
& $VPy -m pip install soundfile

# 4) Smoke: import + cuda (NOT a render -- that downloads ~weights on first use).
& $VPy -c "import torch; from dia.model import Dia; print('dia OK -- torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  setx OTR_ENABLE_DIA 1"
Write-Host ("  (optional) setx OTR_DIA_VENV " + '"' + $VPy + '"')
Write-Host "  Restart ComfyUI. On the canonical workflow set node 80 voice_bank=default"
Write-Host "  and node 81 engine=dia, then queue a small cast."
Write-Host "  (optional quality upgrade) add config\dia_ref_transcripts.json keyed by"
Write-Host "  reference WAV basename to enable transcript-conditioned cloning."
