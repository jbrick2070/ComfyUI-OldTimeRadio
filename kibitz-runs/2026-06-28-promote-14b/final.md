# Promote HuMo-14B (humo_14B_169) -- FINAL build-ready plan (kibitz r1->r4 CONVERGED)

Panel: Codex (gpt-5.5/high) + Claude grounded judge, r1->r4. (Antigravity unavailable: `agy -p`
1.0.13 needs a console DC can't give; the SDK path needs GEMINI_API_KEY not in the DC env.)
Operator decision: 14B fp8 won the bakeoff 100% -> promote it, thin-headroom ACCEPTED. The arc
proved this is a MULTI-NODE per-sub-role ROUTING feature (ROUTE-A), NOT a profile flip. For a
CODER window; operator-gated; SAME-change workflow+code.

## ONE operator input still required before coding
Commit the EXACT engine ids (each must pass `engine_fits_role` for its role):
- `character_video_model = humo_14B_169`  (decided)
- `scene_broll_video_model = <exact id>`   (operator pick; must fit scene_broll inputs)
- `background_abstract_video_model = <exact id>` (operator pick; bg supplies only text_prompt)
NOTE: the current saved `other_beats_video_model="visualizer (16:9)"` is NOT a valid answer for
scene/background (visualizer requires audio_ref; eng_visualizer.py:40-49). Pick role-valid ids.

## The change (ROUTE-A, grounded slices)
1. **Shared role->video-slot helper** (one module/function) used by ALL FOUR current duplicate
   maps so they can't diverge: `OTR_ShotLock` (otr_shot_lock.py:708-715), `OTR_ImageDirector`
   (otr_image_director.py:156-165), `OTR_ImageGenDispatcher` (otr_image_gen_dispatcher.py:
   280-289), `OTR_VideoDirector._role_aspects` (otr_video_director.py:306-325). New per-role
   profile keys: `character_video_model` / `scene_broll_video_model` /
   `background_abstract_video_model`. Image side MUST update too, else the dispatcher
   (visualizer.accepts_still=False) SKIPS the character still HuMo requires (eng_humo.py:331-334).
2. **Profile schema + applier:** `apply_profile` flattens only role/slot/features/seed_policy
   (_otr_workflow_apply.py:428-476); the trusted video-widget allowlist names only the old 3
   Director widgets (:139-141,226-234). Extend the schema + `widget_mapping.json` +
   `_VIDEO_DIRECTOR_WIDGETS` + the mapping/profile tests TOGETHER. Migration note for the old
   `role_overrides.other_beats_visual` key (16gb_full, 8gb_lite, cpu_floor still use it).
3. **Director widgets (positional-safe):** add the 3 new role widgets as OPTIONAL widget-backed
   fields AFTER `custom_models_json` and BEFORE forceInput `gate_in` (serialized order
   _otr_workflow_apply.py:172-204; director widgets otr_video_director.py:203-219) so they
   APPEND to node 87 `widgets_values` (BUG-LOCAL-097, no mid-list insert); append the same
   values to the workflow node 87; run the widget-count/order audit.
4. **Frame cap (tier-specific + exact-fit):** `HUMO_14B_SAFE_RENDER_FRAMES` as a CLASS OVERRIDE
   on `HuMo14BLandscapeEngine` ONLY (eng_humo.py:541-563) -- base `humo`/`humo_1.7B` stay
   uncapped. Render at the cap; before encode, TRIM when over + mirror-extend when short
   (extend_frames_to_target only extends, wrapper_bridge.py:457-459) so
   `frame_count == target_frame_count` (else manifest mismatch + composite holds last frame).
5. **Acceptance (the gates don't see humo_14B_169 today):** update `_episode_facts` to count
   `humo_14B_169` and assert those rows are `role==character_video` (render_driver.py:2030-2047).
   Make the soak gate UNCONDITIONAL: render_shot disables fallbacks (render_driver.py:1468-1500)
   but `assert_soak_ok` + test_video_render_driver.py:74-121 still demand an OOM->still_kenburns
   trail -> retire run_gpu_soak as a gate OR update it + the test to the no-fallback/14B
   invariant. Add an e2e routing test (Director->ShotLock: every shot's engine_id fits its role;
   ImageGenDispatcher keeps HuMo stills) + assert `video_policy["aspects"]` has per-role entries
   for character_video/scene_broll/background_abstract (MetaBrief still-sizing,
   otr_meta_brief_image_prompt.py:150-164).
6. **VRAM safety (settled):** existing post-decode `reclaim_idle_models` + single-resident AS-3
   lease + the 14B frame cap. Two-stage pre-sampler evict CUT (~217 MB, not worth the surface).
   node 92 = single/soak parity only; episode renders from the ShotLock ledger.

## VERIFY-AT-BUILD checklist (Codex r4)
- Final scene/background ids pass `engine_fits_role` for their exact roles.
- node 87 widget order: saved names stay an ordered prefix/subsequence of live INPUT_TYPES; no
  mid-list insert. Run OTR_WorkflowValidator + JSON round-trip + link integrity + wired
  input-name audit + widget-count/order audit.
- Only `HuMo14BLandscapeEngine` gets the cap; `humo`/`humo_1.7B` uncapped.
- Rendered HuMo14B clips: `frame_count == target_frame_count` after trim/extend.
- Live episode histogram: `humo_14B_169 > 0` ONLY on `role==character_video`.
- HuMo14B preflight: OTR_ENABLE_HUMO + ckpt path + wrapper node classes; no OOM at a
  representative AND a max-cap beat.

## Invariants
Single resident <=14.5 GB is technically violated (~15.9 GB) -- thin-headroom ACCEPTED by the
operator, bounded by single-resident + the beat cap; make the acceptance explicit, not silent.
UTF-8 no BOM; SFW; workflow+code SAME change; suite + Bug Bible + B7 per green chunk; commit+push
v2.0-alpha; prod/main + tags GATED.

## Build order
operator picks scene/bg ids -> shared role->slot helper (video+image) -> per-role profile keys +
schema/applier/widget_mapping/_VIDEO_DIRECTOR_WIDGETS + migration note -> append Director widgets
+ workflow same change + widget audit -> HuMo14B frame-cap override + exact-fit -> acceptance
counters + soak-gate fix + e2e routing test + aspects test -> validator/round-trip/audits ->
suite + Bug Bible + B7 -> humo_14B_169 preflight -> live episode (histogram on character rows,
OBS publish, no OOM at representative + max-cap) -> operator eyeball -> commit per green chunk.
