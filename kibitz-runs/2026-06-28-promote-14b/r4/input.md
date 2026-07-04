# Promote HuMo-14B -- r3-hardened WIRING plan (Codex + Claude; grounded). ROUTE-A.

r3 panel: Codex (gpt-5.5/high) + Claude. Route-B CUT (no clean config source; ShotLock
special-case = second routing language). ROUTE-A (per-sub-role routing) is the path. Scope is
an ARCHITECTURE feature across video+image+profile, NOT a profile flip. All MUST-FIX grounded.

## The change = per-sub-role VIDEO+IMAGE routing (one shared map; do not let nodes diverge)
1. New profile keys (per-role): `character_video_model` / `scene_broll_video_model` /
   `background_abstract_video_model`. Set `character_video_model = humo_14B_169`; scene +
   background -> engines that ACTUALLY pass `engine_fits_role` (role_compat.py:107-131) for
   their inputs -- NOT "visualizer" for background (visualizer needs `audio_ref`,
   eng_visualizer.py:45; `background_abstract` has only `text_prompt`, role_compat.py:69-72).
   Verify what currently renders scene/background and keep those role-valid engines.
2. Update BOTH the VIDEO and IMAGE role->slot maps via ONE shared helper: `OTR_VideoDirector`
   + `OTR_ShotLock` (otr_shot_lock.py:708-780) AND `OTR_ImageDirector`
   (otr_image_director.py:156-196) + `OTR_ImageGenDispatcher` (otr_image_gen_dispatcher.py:
   280-341), which today all map character/scene/background -> `other_beats_video_model`. If
   not fixed on the image side, the dispatcher (visualizer.accepts_still=False,
   eng_visualizer.py:45-49) SKIPS the character still that HuMo REQUIRES (eng_humo.py:331-334).
3. Profile schema + applier: `apply_profile` flattens only role_overrides/slot_overrides/
   features/seed_policy (_otr_workflow_apply.py:428-476) and the trusted video-widget allowlist
   names only the OLD three Director widgets (:139-141,226-234). Extend the schema + 
   `config/profiles/widget_mapping.json` + `_VIDEO_DIRECTOR_WIDGETS` + the mapping/profile
   tests TOGETHER.
4. Workflow widgets: APPEND the new Director widgets at the END (widgets_values is positional
   -- BUG-LOCAL-097/CLAUDE.md S0; INPUT_TYPES order otr_video_director.py:148-224); update
   workflows/otr_scifi_16gb_full.json same change; run the widget-count vs INPUT_TYPES audit.

## Frame cap (tier-specific + exact-fit)
- `HUMO_14B_SAFE_RENDER_FRAMES` as a CLASS OVERRIDE on `HuMo14BLandscapeEngine`
  (eng_humo.py:541-563) -- NOT the base class (must not cap humo_1.7B). Render at the cap.
- Exact-fit before encode: `extend_frames_to_target` only mirror-extends when SHORT and
  returns unchanged when target<=n (wrapper_bridge.py:457-459); HuMo quantizes up to 4n+1
  (min33/max177). Add a TRIM-when-over step so `frame_count == target_frame_count` exactly,
  else the manifest count mismatches + the composite holds the last frame
  (otr_silent_composite.py:237-262).

## Acceptance (fix the gates, they don't see humo_14B_169 today)
- `_episode_facts` counts exact "humo" only (render_driver.py:2030-2047) -> update to count
  `humo_14B_169`; assert every such row is `role==character_video`.
- `assert_soak_ok` still expects fallback decisions but render_shot disables fallbacks
  (render_driver.py:1468-1495,2069-2107) -> fix if run_gpu_soak stays a gate.
- END-TO-END routing test (not a Director-only test): Director policy -> ShotLock shots ->
  every shot's `engine_id` fits its `role` + ImageGenDispatcher keeps the required HuMo stills.

## Carried / settled
- VRAM safety = existing post-decode reclaim + single-resident lease + the 14B frame cap;
  two-stage pre-sampler evict stays CUT (~217 MB, not worth the surface).
- Profile<->workflow EXACT-match fixtures updated to new truth (test_capability_profiles.py
  :176-205, test_workflow_apply.py:111-117), not weakened.
- node 92 (OTR_VideoRenderBatch.engine) = single/soak parity only; episode renders from the
  ShotLock ledger.

## OPERATOR product call (gate)
Confirm: character beats = humo_14B_169 (the 14B talking head); scene_broll +
background_abstract = which role-valid engines (the current ones). This is a routing/product
decision + a real multi-node feature -- size it as such, not a flip.

## Build order
shared role->slot map (video+image) -> new profile keys + schema/applier/widget_mapping/
_VIDEO_DIRECTOR_WIDGETS -> append Director widgets + workflow same change + widget audit ->
HuMo14B frame-cap class override + exact-fit trim/extend -> acceptance counters + e2e routing
test -> OTR_WorkflowValidator + round-trip + audit -> suite + Bug Bible + B7 -> humo_14B_169
preflight -> live episode (histogram humo_14B_169>0 on character rows, OBS publish, no OOM at
representative + max-cap beat) -> operator eyeball -> commit per green chunk.
