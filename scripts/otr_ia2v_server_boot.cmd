@echo off
rem ia2v proof/production server boot: _otr_soak_server_launch.cmd with the
rem STANDARD engine env the full pipeline needs (the bare soak launcher sets
rem NONE of these -- booting without them fails the z_image portrait mint:
rem load_unet FileNotFoundError, caught proof8 2026-07-02). Keep this env in
rem sync with scripts\_otr_overnight_420_boot.cmd.
rem Usage: _otr_ia2v_server_boot.cmd "<server-log-path>"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
set OTR_ENABLE_LTX_AV=1
rem FLAT name -- the GGUF lives at <models>\unet\<name> (C:\ComfyUI-Models via
rem the extra-paths yaml); an ltx2\ subfolder prefix breaks the resolve (that
rem prefix belongs to the LORA path only). Caught proof8 attempt 2.
set OTR_LTX_AV_UNET=ltx-2.3-22b-dev-Q3_K_M.gguf
set OTR_LTX_AV_RECIPE=
set OTR_ZIMAGE_UNET=z_image_turbo_nvfp4.safetensors
set OTR_ZIMAGE_CLIP=qwen_3_4b_fp8_mixed.safetensors
set OTR_ZIMAGE_VAE=ae.safetensors
call "%~dp0_otr_soak_server_launch.cmd" %1
