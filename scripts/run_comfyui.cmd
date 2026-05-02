@echo off
REM run_comfyui.cmd -- launch ComfyUI Desktop with HF_HOME inherited.
REM
REM BUG-LOCAL-003 fix (2026-05-02). The Electron-bundled ComfyUI Desktop
REM does not read user-scope env vars from HKCU\Environment when launched
REM from the Start menu shortcut or a child shell that lacks them. This
REM helper reads HF_HOME + HUGGINGFACE_HUB_CACHE from HKCU\Environment via
REM PowerShell, exports them into this cmd shell, then launches the
REM ComfyUI.exe. Child Python processes inherit the env correctly.
REM
REM Without this, Mistral-Nemo (and any other HF model) fails to resolve
REM from the canonical C:\ComfyUI-Models\huggingface cache and falls back
REM to ~/.cache/huggingface (typically empty), triggering offline-mode
REM "no safetensors found" errors deep in the LLM phase.

setlocal

REM Read HF_HOME from HKCU\Environment via PowerShell.
for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('HF_HOME','User')"`) do set HF_HOME=%%V
for /f "usebackq delims=" %%V in (`powershell -NoProfile -Command "[Environment]::GetEnvironmentVariable('HUGGINGFACE_HUB_CACHE','User')"`) do set HUGGINGFACE_HUB_CACHE=%%V

REM Fall back to the canonical OTR cache root if user-env was empty.
if "%HF_HOME%"=="" set HF_HOME=C:\ComfyUI-Models\huggingface
if "%HUGGINGFACE_HUB_CACHE%"=="" set HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub

REM Optional: enable offline mode so HF doesn't probe the network. Comment
REM out the next two lines if you intentionally want HuggingFace to fetch
REM updates.
set HF_HUB_OFFLINE=1
set TRANSFORMERS_OFFLINE=1

echo HF_HOME=%HF_HOME%
echo HUGGINGFACE_HUB_CACHE=%HUGGINGFACE_HUB_CACHE%
echo HF_HUB_OFFLINE=%HF_HUB_OFFLINE%
echo Launching ComfyUI Desktop...

start "" "C:\Users\jeffr\AppData\Local\Programs\ComfyUI\ComfyUI.exe"

endlocal
