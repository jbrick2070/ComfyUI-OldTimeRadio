# Claude anchor review -- r1 (arc/coverage) -- HYGIENE_FIX_PLAN

VERDICT: fix is correct + minimal; 1 must-fix coverage gap (peer audit) + 2 confirms. Grounded.

## CONFIRMED
- ROOT CAUSE exact: `otr_scene_aware_scopes.py:537` writes `out_path = os.path.join(
  tempfile.gettempdir(), f"otr_scopes_{key}_{ts}.mp4")` and `:541 return {"result": (out_path,)}` --
  NO delete. The file persists in the ambient TEMP dir. On a non-launcher server (TEMP = system Temp),
  it orphans in `%LOCALAPPDATA%\Temp` and the gate (`_otr_soak_capstone.py:225-276`) flags it. CONFIRMED.
- THE FIX: route line 537 to `nodes/_otr_paths.otr_shared_tmp_dir()` (the `otr/episodes/_shared/tmp`
  tier the OH-3 janitor sweeps) instead of `tempfile.gettempdir()`. The gate only inspects system Temp,
  so this alone makes it pass -- AND it is hygiene-correct (the transient lands in the swept tier),
  independent of whether a launcher repointed TEMP. CONFIRMED the authority exists (_otr_paths:254).

## MUST-FIX
M1. PEER AUDIT (coverage): the SAME bug class exists at `nodes/video_engine.py:2201`
    (`os.path.join(tempfile.gettempdir(), "otr_video_audio.wav")`) and `nodes/scene_sequencer.py:1275`
    (gettempdir wav dir). The overnight soak only flagged the scopes file because that is what node-94
    produced on the live in-process path; video_engine:2201 may be a legacy/unused path (verify it is
    NOT on the live render before deciding to fix or leave). Any PERSISTENT `otr_*` system-temp writer
    on a live path is a latent gate-failure -- route them through the path authority too, OR confirm
    they're off-path / self-deleting. The `mkdtemp(prefix="otr_*")` callers (otr_silent_composite:751,
    rtx_upscale:653, eng_mesh_stage) clean up via rmtree in a finally -> they don't persist -> leave.

## CONFIRM (not gaps)
- The gate is CORRECT (the node was wrong); do NOT relax the gate. Routing the scratch is the fix.
- No determinism / IS_CHANGED / content impact (a transient OUTPUT path, not a seed or cache key); no
  workflow-JSON change (pure node-internal). Master audio untouched (scopes is video-only).

## VERIFY-AT-BUILD
Whether otr_shared_tmp_dir() resolves cleanly when called from node-94's context (it should -- same
authority node-92 uses for state); fail-soft loud if not (mirror otr_engine_tmp_mp4), never silently
fall back to system temp on the production path.
