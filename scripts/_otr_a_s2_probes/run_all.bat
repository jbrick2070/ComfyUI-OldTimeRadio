@echo off
REM A-S2 de-risk probes -- run on the RTX 5080 from the repo root.
REM Each prints PASS or NO-GO and sets errorlevel. Paste the 5 result lines
REM back into the next window. GO (all PASS) -> A-S3/CW-4 render path.
setlocal
set PY=C:\Users\jeffr\Documents\ComfyUI\.venv\Scripts\python.exe
set HERE=%~dp0
echo ============================================================
echo A-S2 DE-RISK PROBES (operator / 5080)
echo ============================================================
"%PY%" "%HERE%probe_b_coldimport_dep_pilot.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_a_flux_load_teardown_leak.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_c_silent_mux_audio_hash.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_d_vram_boundary_failclosed.py"
echo ------------------------------------------------------------
"%PY%" "%HERE%probe_e_sidecar_vram_free.py"
echo ============================================================
echo Done. Review each PASS / NO-GO line above.
endlocal
