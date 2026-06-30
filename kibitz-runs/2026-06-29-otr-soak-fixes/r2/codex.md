VERDICT: no. The plan still leaves S-F/S-A decisions unresolved and several steps contradict current workflow/code interfaces.

MUST-FIX BEFORE BUILD:
1. [S-F] Injection is not build-specified and one input list is wrong. Workflow node 91 has inputs `script_json`, `image_policy_json`, `image_prompts_json`, `gate_in`, `episode_id`; it does not take `audio_done` (workflows/otr_scifi_16gb_full.json node 91). Node 90 owns `audio_done`; node 92 takes `patched_ledger_json`, `master_audio_path`, `image_done`. Concrete fix: choose A or B now. If A, define exact API prompt surgery: add/replace constant STRING producer nodes for node 91/92 sockets and disconnect upstream writer/audio/image links so upstream nodes are not dependencies. If B, define the fixture-loader node outputs and wire them into production JSON in the same change.

2. [S-F] “Invalidate writer/audio node cache” is not an implemented API in the shown harness. `scripts/otr_api.py` converts UI workflow links/widgets into `/prompt` inputs; it has widget patching and queue helpers, not cache invalidation (scripts/otr_api.py:541-656, 851-866). Concrete fix: make non-execution structural by severing links to writer/audio nodes or by a fixture-loader node, then prove via trace that those nodes were not dependencies.

3. [S-A] The HuMo frame-cap statement is inaccurate as written. `_HUMO_14B_SAFE_RENDER_FRAMES = 49` exists, but only `HuMo14BLandscapeEngine` sets `safe_render_frames`; base `HuMoEngine` and `HuMo17BEngine` leave it `None` and still cap via `_HUMO_MAX_FRAMES = 177` (nodes/_otr_video_engines/eng_humo.py:53-61, 102-108, 352-358, 374-375, 489-505, 564-585). Concrete fix: restate the target as composite-side fill for any `clip.frame_count < target_frame_count`, or explicitly cap/exact-fit each engine class intended.

4. [S-A] “Replace `tpad=clone` with boomerang/loop extender” is too blunt and will affect floor slices unless scoped. Regular clip segments currently call `emit(..., loop=False)` even after underrun warning; only the credits-tail path sets `loop=True` (nodes/otr_silent_composite.py:325-339, 360-368, 395-411, 645-649). Concrete fix: in `plan_timeline_segments`, set `loop=True` only for real clip rows where `exists && path && frame_count > 0 && frame_count < target_frame_count`; keep `tpad` as safety for non-loop/floor sources.

5. [S-E] Deleting fallback constants/functions will break live call sites and tests unless the call surface is migrated first. `run_real_episode` still defaults to `make_fallback_of()` and `render_single` builds `fb = make_fallback_of()` (nodes/_otr_video_engines/render_driver.py:1737, 2217), while tests import/assert `FLOOR_NAMES`, `UNIVERSAL_FLOOR`, `SYNTH_FALLBACKS`, and `make_fallback_of` (tests/test_video_character_3d.py:363-369; tests/test_video_render_driver_additive.py:77-82). Concrete fix: either keep compatibility stubs marked dead/no-op until tests migrate, or remove all call sites/imports/tests in the same commit.

6. [S-E] Removing `allow_auto_fallback` is a workflow/widget migration, not just code deletion. The live node 87 still has an `allow_auto_fallback` input and the positional widget value `false` in `widgets_values` (workflows/otr_scifi_16gb_full.json node 87); code signature also requires it and writes it into policy JSON (nodes/otr_video_director.py:216, 278-284, 340-342). Concrete fix: update `INPUT_TYPES`, `direct()` signature, policy schema consumers/tests, and node 87 `inputs`/`widgets_values` in one JSON migration with the widget audit.

7. [S-E] The proposed durable per-beat video stamp targets the wrong merge layer. `_merge_with_disk` preserves top-level keys and row arrays `lines/clips/music`; it does not merge `video.shots` keyed by `shot_id` (nodes/production_ledger.py:1191-1283). Adding all `video` to `TOP_PRESERVE` risks preserving stale shot plans wholesale. Concrete fix: either extend existing `meta.render_engines` (already saved by nodes/otr_video_render_batch.py:26-50) with per-beat details, or add a keyed merge for `video.shots` by `shot_id`.

8. [S-B/S-E] Recipe/quant stamping cannot be implemented from current clip data. `eng_ltx_av` logs recipe/unet but returns only `out_path`, `frame_count`, and `vram_peak_mb`; `_clip_from_raw` preserves only `vram_peak_mb` (nodes/_otr_video_engines/eng_ltx_av.py:621-627, 691-694, 776-788). Concrete fix: return `recipe`, `unet`, `lora`, `quant`, `canvas`, `audio_source`, and `phase` in raw/canonical clip, then thread them into `build_clip_manifest`.

SHOULD-FIX:
1. [S-D] Put the schema-name unwrap on `RadioEditPlan`, not the shared tolerant core. The shared helper explicitly says alias drift is schema-owned and only clamps overlong strings after `schema.model_validate` (nodes/_otr_structured_call.py:322-351). Concrete fix: add `@model_validator(mode="before")` on `RadioEditPlan` that unwraps exactly `{"RadioEditPlan": {...}}` and rejects ambiguous/multi-key wrappers.

2. [S-A] Legibility guard has no data contract. `build_clip_manifest` includes `init_image`, `init_source`, and `init_image_used`, but not a universal source-still path or quality fields (nodes/_otr_video_engines/render_driver.py:2000-2015). Concrete fix: define manifest fields for `sharpness_ratio`, `freeze_score`, `quality_status`, and `fail_reason`, plus thresholds and the exact hard-fail location.

3. [S-E] Unregistering `abstract`, `still_motion`, and `station_card` is wider than fallback cleanup. They are registered engines with default roles and capability rows (nodes/_otr_video_engines/cheap_families.py:165-190; nodes/_otr_video_engines/registry.py:127-133). Concrete fix: separate “remove fallback use” from “remove selectable engines”; if removing them, update defaults/profiles/tests explicitly.

4. [BUG-411] “May also be done” is not a coding task. Concrete fix: convert the remaining suffix/seed checks into a checklist with current truth from `nodes/otr_meta_brief_image_prompt.py`, then implement only missing items.

OPTIONAL / NICE-TO-HAVE:
1. [S-B] Replace the stale `13688` comments with links to generated bakeoff manifests, not just “see logs,” so future reviews have a stable artifact.

CUT THESE (over-engineering):
1. [S-A] Cut sharpness-ratio gating from the first clip-fill commit. Frame-count exactness plus `freezedetect` catches the underrun/freeze bug with less new data plumbing; add sharpness after the motion contract is green.

2. [S-F] Cut the fixture-loader node option unless production needs fixture mode. API prompt surgery/constant producers are enough for a test-only accelerator and avoid adding a production node that exists only to bypass production stages.