<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The core wiring is solid, but synthetic-beat audio slicing is broken, dark-lane fallbacks bypass their chains, and ShotLock lacks the validation the docs claim it has.

MUST-FIX BEFORE BUILD:

1.  [PASS04-1] **Synthetic Music Beat Slicing Starvation**: `render_driver.py` `build_request_from_shot` (lines 344-347) attempts to slice the master audio using `start_s = line.get("start_s")`. For the synthetic opening-music beat, `line` is empty, `start_s` is `None`, and slicing is skipped. `ltx_av_music` will starve and fail-closed.
    *Fix*: Update lines 344-345 to fallback to the shot's timing ONLY for the new engine (preserving byte-identical requests for existing engines):
    `start_s = line.get("start_s") if line.get("start_s") is not None else (shot.get("start_s") if shot.get("engine_id") == "ltx_av_music" else None)`
    `dur_s = line.get("dur_s") if line.get("dur_s") is not None else (shot.get("dur_s") if shot.get("engine_id") == "ltx_av_music" else None)`
2.  [PASS04-2] **Dark Lane Fallback Bypass**: If `OTR_ENABLE_LTX_AV` is off, the engines are unregistered. `render_driver.py` `make_fallback_of` (line 100) defaults unregistered engines straight to `UNIVERSAL_FLOOR` (`still_kenburns`), completely bypassing `humo` and `ltx_video`.
    *Fix*: Add the new engines to `SYNTH_FALLBACKS` in `render_driver.py` (line 53): `{"hunyuan3d_talk": "humo", "ltx_av_talk": "humo", "ltx_av_music": "ltx_video"}`.
3.  [PASS04-2] **Phantom ShotLock Validation**: `registry.py` docstring claims `OTR_ShotLock` calls `assert_usable` to fail closed. It does not; `otr_shot_lock.py` never calls it. If an engine is picked but the flag is off, Director bypasses validation (because it's not in `known`), ShotLock blindly stamps it, and the episode relies entirely on `render_driver` to fail and walk the chain.
    *Fix*: Either implement the `assert_usable` check in `otr_shot_lock.py` `build_execution_plan` (swapping to `default_engine_for_role` on failure), or correct the `registry.py` docstring to state that validation happens at render-time.
4.  [PASS04-4] **Missing Identity Stamps**: `build_clip_manifest` relies on `clip.get("engine_id")`. If the new adapters don't stamp this, the acceptance grep cannot prove identity.
    *Fix*: `eng_ltx_av.py`'s `canonicalize` method MUST inject `"engine_id": self.name` into the returned `CanonicalClip` dict.

SHOULD-FIX:

1.  [PASS04-3] **Orphaned Groups Leak**: `resolver.py` claims the in-render resolver calls `prune_orphaned_groups` on a fallback-restamp. `render_driver.py` `run_episode` (lines 491-512) appends decisions and restamps shots, but NEVER calls `prune_orphaned_groups`. Execution groups leak orphaned providers.
    *Fix*: Call `validate_execution_groups(..., prune_orphans=True)` inside `run_episode` if `decisions` is non-empty.
2.  [PASS04-5] **Force Map Flag Dependency**: `OTR_FORCE_ENGINE_MAP` parsing (`render_driver.py:560`) raises `ValueError` if the engine is unregistered. This means the operator CANNOT force the new lane for the M4 smoke test if the flag is off.
    *Fix*: Either ensure the flag is ON for the smoke test, or bypass the `is_registered` check for forced overrides.

OPTIONAL / NICE-TO-HAVE:

*   **Restamp Wording**: To ensure the exact wording for the LOUD restamp grep, `eng_ltx_av.py` `assert_usable` / `prepare` should raise `EngineUnusable` with exactly:
    (a) `"aspect change landscape -> pillarbox"`
    (b) `"fallback ltx_av_music -> ltx_video"`
    (c) `"pad-tail > 2s"`
*   **Portrait Supply (Focus 6)**: Announcer beats reliably get an `init_image` via `ledger['images']` (synthetic non-cast portrait, `portrait_ledger.py:175`). If missing, `ltx_av_talk` fails closed (as required), the driver walks to `humo`, which *also* fails closed (missing image), terminating safely at `still_kenburns`. No changes needed, behavior is correct.
*   **Seeds (Focus 7)**: `request_seed` correctly derives from `render_request_hash` deterministically (`render_driver.py:309`). C7 env overrides (`OTR_CAST_SEED`) are safely irrelevant here. No changes needed.

CUT THESE (over-engineering):
1.  None. The wiring is remarkably lean.

[ASSUMPTION] `eng_ltx_av.py` will correctly raise `EngineUnusable` for the aspect ratio and pad-tail violations, as the logic is not visible in the provided grounding but is required by the plan.