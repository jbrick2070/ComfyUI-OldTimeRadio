@echo off
REM Subproject B de-risk spikes -- run on the RTX 5080 from the repo root.
REM Keystone-FIRST: selftest -> b (pre-screen) -> c (ARKit WRAP keystone) THEN
REM a (cu128 CUDA-ext) -> d (A2F-3D onset) -> e (render-spawn VRAM). If c is a
REM NO-GO, STOP -- character_3d does not ship (HuMo-2D stays the v1 path).
REM Point OTR_B_SIDECAR_PYTHON at the cu128 sidecar venv for probes a + e.
setlocal
set PY=C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
if "%OTR_B_SIDECAR_PYTHON%"=="" (set SIDE=%PY%) else (set SIDE=%OTR_B_SIDECAR_PYTHON%)
set HERE=%~dp0
echo ============================================================
echo B DE-RISK SPIKES (operator / 5080) -- keystone first
echo ============================================================
"%PY%" "%HERE%selftest_harness.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_b_manifold_prescreen.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_c_arkit_wrap.py"
echo ------------------------------------------------------------
"%SIDE%" "%HERE%probe_a_cudaext_compile_load_sm120.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_d_a2f3d_onset_variance.py"
echo ------------------------------------------------------------
"%SIDE%" "%HERE%probe_e_sidecar_vram_free_3d.py"
echo ============================================================
echo Done. Review each PASS / NO-GO line above. Keystone = probe c.
endlocal
