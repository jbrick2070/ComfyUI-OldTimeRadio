# SOAK HYGIENE FIX -- CONVERGED (kibitz r1: Claude Code + Codex + Antigravity, grounded)

Scope: the SYSTEM-TEMP HYGIENE failures only (the 11 legs that rendered + published to obs but tripped
the gate). The 3 genuine render failures (ltx_audio_in floored, mesh_stage, wan_i2v) are OUT of scope
here (separate). No shim; the gate is CORRECT and stays strict (panel consensus).

## THE FIX
1. `otr_scene_aware_scopes.py:537` -- write the scopes intermediate to `_otr_paths.otr_shared_tmp_dir()`
   (the swept `otr/episodes/_shared/tmp` tier) instead of `tempfile.gettempdir()`, with
   `os.makedirs(..., exist_ok=True)` at the allocation point (otr_shared_tmp_dir returns a path; ensure
   it exists). DO NOT delete in the producer -- node 93 `OTR_PostUpscaleProcgenBlend` consumes
   `scopes_mp4_path` downstream (otr_post_upscale_procgen_blend.py:903-912) and the OH-3 janitor owns
   cleanup of the _shared/tmp tier. Lazy import inside the function (V-12 cold-import discipline).
2. PEER FIXES (the same otr_*-named system-temp class, grounded): route + guard the two clear writers:
   - `video_engine.py:2201` `otr_video_audio.wav` -> `otr_shared_tmp_dir()` + wrap its `os.remove` in a
     `finally` (today the cleanup at :2404 is not in a finally -> leaks on ffmpeg failure).
   - `scene_sequencer.py:1275` gettempdir master-wav fallback -> `otr_shared_tmp_dir()`.
   - `eng_mesh_stage.py:607` selftest `mkdtemp(prefix="otr_mesh_selftest_")` -> wrap in try/finally
     `shutil.rmtree(..., ignore_errors=True)` (leaks on Blender/validation failure).
   CLASSIFY as intentional / off-render (leave): `gpu_residency.py:51` (env-gated OTR_GPU_LEASE_DIR
   lock), `_otr_openrouter_backend.py:1424` (`__main__` self-test only). The cleaned `mkdtemp` callers
   (otr_silent_composite:751 rmtree@:780, rtx_upscale:653 clean@:699-706) self-delete -> leave.
3. REGRESSION GUARD (the existing `test_engine_tmp_in_tree.py` only scans `_otr_video_engines/*.py`):
   add a TOP-LEVEL node-hygiene test that scans ALL workflow-reachable `nodes/**/*.py` and FAILS on a
   `tempfile.gettempdir()` joined with an `otr`-prefixed literal (with a small allowlist for the
   classified-intentional paths) + a focused `SceneAwareScopes` assertion that its out_path resolves
   under the OTR tmp tier, never system temp.

## ACCEPTANCE
- A scopes render leaves NO `otr_*` entry in `%LOCALAPPDATA%\Temp` regardless of the ambient TEMP env;
  the intermediate lands in `otr/episodes/_shared/tmp`. The downstream blend still reads it (scopes
  layer still appears).
- Suite + Bug Bible + B7 green; the new hygiene scan green; master audio byte-identical (scopes/peers
  are video/scratch only). No workflow-JSON change.
- LIVE confirm: a single render leg (or the previously-failed scopes legs) re-run via the soak passes
  the capstone hygiene gate (pytest alone does not exercise the gate -- Antigravity).

## CONVERGENCE
r1 (3 agents + grounded anchor) agreed on the shape; the only corrections to the input plan were
"don't delete in the producer" (consumer + janitor own cleanup) and "the regression test must scan
top-level nodes, not just _otr_video_engines". No wiring/JSON change -> r3 is a no-op; r2 coding is the
concrete fixes above; converged. BUILD-READY.
