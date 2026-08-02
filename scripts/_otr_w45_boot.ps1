# Boot ONE headless server with EVERY engine the 45-word campaign needs.
#
# The .cmd launcher's %2 lanes (FLOOR/HUMO/LTX/WAN) are mutually exclusive by
# design -- they were written for single-engine bakeoffs. A six-engine campaign
# needs them all at once, so the flags are set HERE, in the caller's process
# environment, which the launcher explicitly supports ("harnesses that need a
# special env must pass it explicitly ... so the server log and command line
# remain auditable"). HUMO is passed as the lane because it is the only branch
# that does not clear OTR_ENABLE_HUMO, and it leaves the LTX/WAN flags alone.
#
# NOTE what is deliberately ABSENT: GEMMA4_12B_MAX_NEW_TOKENS. Until 2026-08-01
# the story writer only worked because that var was exported at boot -- a dead
# channel one forgotten restart away from silently reintroducing the failure.
# DEFAULT_OUTPUT_TOKENS_CAP is now 4096 in code, so this boot proves the fix
# stands on its own.
$ErrorActionPreference = 'SilentlyContinue'

$REPO = 'C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio'
$LOG  = Join-Path $REPO 'tmp\_w45_server.log'
$LAUNCH = Join-Path $REPO 'scripts\_otr_soak_server_launch.cmd'

# --- 1. Reset (section 4): selective kill, never a blanket python kill. -------
Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'ComfyUI\\main\.py|ComfyUI/main\.py' } |
    ForEach-Object { "killing server pid=$($_.ProcessId)"; Stop-Process -Id $_.ProcessId -Force }

Get-NetTCPConnection -LocalPort 8000 -State Listen |
    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }

$deadline = (Get-Date).AddSeconds(90)
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 3
    $listen = @(Get-NetTCPConnection -LocalPort 8000 -State Listen)
    $vram = [int](nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    if ($listen.Count -eq 0 -and $vram -le 2600) { break }
}
"port 8000 free: $((@(Get-NetTCPConnection -LocalPort 8000 -State Listen)).Count -eq 0)"
"vram before boot: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"

# --- 2. Every engine the campaign submits. -----------------------------------
$env:OTR_ENABLE_HUMO        = '1'   # humo
$env:OTR_ENABLE_LTX_VIDEO   = '1'   # ltx_video, ltx_8gb
$env:OTR_ENABLE_LTX_I2V     = '1'
$env:OTR_ENABLE_LTX_AV      = '1'   # ltx_audio_in
$env:OTR_ENABLE_WAN_TI2V    = '1'   # wan_ti2v AND fastwan_8gb (subclass)
$env:OTR_WAN_TI2V_CKPT      = 'C:\ComfyUI-Models\diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf'
$env:OTR_WAN_TI2V_VAE_NAME  = 'wan2.2_vae.safetensors'

# Fresh entropy per episode: cast/style/fable2 seeds stay UNSET (no OTR_C7).
Remove-Item Env:\OTR_C7 -ErrorAction SilentlyContinue
Remove-Item Env:\GEMMA4_12B_MAX_NEW_TOKENS -ErrorAction SilentlyContinue

# --- 3. Boot. -FilePath, never cmd.exe /c (the /c two-quoted-token rule eats
#        the outer quotes -> mangled path -> zero log output).
Start-Process -FilePath $LAUNCH -ArgumentList "`"$LOG`"", 'HUMO' -WindowStyle Hidden

# --- 4. Boot is ~20s. If it "hangs", it has already died -- read the log.
$deadline = (Get-Date).AddSeconds(180)
$up = $false
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 5
    if (@(Get-NetTCPConnection -LocalPort 8000 -State Listen).Count -gt 0) { $up = $true; break }
}
if ($up) {
    "SERVER UP on :8000"
    Select-String -Path $LOG -Pattern '\[launch\]' | ForEach-Object { $_.Line }
} else {
    "SERVER DID NOT COME UP -- last 25 log lines:"
    Get-Content $LOG -Tail 25
    exit 1
}
