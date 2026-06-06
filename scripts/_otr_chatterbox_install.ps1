# OTR Chatterbox Path-B install -- isolated venv so chatterbox's pinned torch /
# numpy NEVER touch the main ComfyUI torch-2.10 / cu130 venv. Run from anywhere.
# After it finishes: setx OTR_ENABLE_CHATTERBOX 1, then RESTART ComfyUI.
$ErrorActionPreference = "Stop"
$Root = "C:\Users\jeffr\Documents\ComfyUI\chatterbox"
$Venv = Join-Path $Root ".venv"
New-Item -ItemType Directory -Force -Path $Root | Out-Null

# 1) Create the isolated venv (Python 3.11 recommended for chatterbox).
if (Get-Command py -ErrorAction SilentlyContinue) { py -3.11 -m venv $Venv } else { python -m venv $Venv }
$VPy = Join-Path $Venv "Scripts\python.exe"

# 2) Install chatterbox. It pins its OWN torch / numpy -- that is the whole point
#    of the isolated venv; do not "fix" the pins against the main venv.
& $VPy -m pip install --upgrade pip
& $VPy -m pip install chatterbox-tts soundfile

# 3) BLACKWELL (RTX 5080, sm_120) CHECK -- verify-at-build. If the smoke below
#    prints cuda=False or the first render errors with an sm_120 / kernel message,
#    reinstall a cu128 torch INTO THIS VENV ONLY (uncomment, pin after testing):
# & $VPy -m pip install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio

# 4) Smoke: import + report torch / cuda (downloads the model on first run).
& $VPy -c "import torch, torchaudio; from chatterbox.tts import ChatterboxTTS; print('chatterbox OK -- torch', torch.__version__, 'cuda', torch.cuda.is_available())"

Write-Host ""
Write-Host "Done. Next steps:"
Write-Host "  setx OTR_ENABLE_CHATTERBOX 1"
Write-Host ("  (optional) setx OTR_CHATTERBOX_VENV " + '"' + $VPy + '"')
Write-Host "  Restart ComfyUI. On the canonical workflow set node 80 voice_bank=default"
Write-Host "  and node 81 engine=chatterbox, then queue a small cast."
