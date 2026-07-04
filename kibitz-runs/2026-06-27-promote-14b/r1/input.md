# Promote HuMo-14B (humo_14B_169) -- coding plan for a coder window (pass00, to harden)

Operator decision 2026-06-27: 14B fp8 won the bakeoff 100%; promote it, ACCEPTING thin VRAM
headroom (it rendered every leg zero-OOM at ~15.9 GB; FINAL_VERDICT.md). This is a PRODUCTION
change (profile + workflow + maybe engine), operator-gated, run by a CODER window. Harden the
plan r1->r4 before building.

## Grounded facts (verify at build)
- `config/profiles/16gb_full.json`: `role_overrides.other_beats_visual="humo_1.7B"`,
  `slot_overrides.video_render_engine="humo_1.7B"` (NOT _169).
- `config/profiles/widget_mapping.json`: `role_overrides.other_beats_visual` ->
  `[OTR_VideoDirector, other_beats_video_model]` (registry "video", value validated against the
  video enable-set); `slot_overrides.video_render_engine` -> `[OTR_VideoRenderBatch, engine]`.
  Applier = `nodes/_otr_workflow_apply.py`; tests `test_capability_profiles.py` +
  `test_workflow_apply.py`.
- The SAVED `workflows/otr_scifi_16gb_full.json` currently has
  `OTR_VideoDirector.other_beats_video_model = "visualizer (16:9)"` and
  `OTR_VideoRenderBatch.engine = "humo_1.7B"` -- DRIFT from the profile (the 2026-06-23
  HuMo-free UI-save). This drift is the cause of the 5 pre-existing suite fails
  (16gb-profile / workflow-structure / audio-wiring pins).
- Episode mode renders from the ShotLock ledger via role routing
  (`otr_video_render_batch.py:127-134`); node 92 `OTR_VideoRenderBatch.engine` is single/soak
  parity ONLY, ignored in episode mode.
- `humo_14B_169` = `HuMo14BLandscapeEngine` (wide 832x480, 14B fp8 + lightx2v distill, 6 steps
  / cfg 1.0 / shift 8); it is in the validated video engine set.
- `eng_humo.render_clip` already does POST-decode `reclaim_idle_models` + the single-resident
  AS-3 lease; the bakeoff's PRE-sampler two-stage evict saved only ~217 MB (Step A), so the
  14B at ~15.9 GB is weight-dominated regardless.

## OPEN QUESTION the plan must resolve (operator + panel)
The profile pins `humo_1.7B` for other_beats, but the saved workflow uses `visualizer (16:9)`
(HuMo-free), and HuMo needs an init_image + audio_ref (audio_driven_face) -- so it is NOT
role-valid for face-less scene b-roll. So WHERE does the 14B go?
  (a) Replace `humo_1.7B` -> `humo_14B_169` wherever the 1.7B was the audio-driven-face pick
      (the role-routed character/announcer/music face beats), reconciling the profile<->workflow
      drift; OR
  (b) the operator wants HuMo back for other-beats generally (revert the HuMo-free visualizer
      UI-save to the 14B). 
This is a creative/routing call -- the plan must state it explicitly, not guess.

## Coding slices (SAME-change profile + workflow per CLAUDE.md S0)
1. Profile: `16gb_full.json` `other_beats_visual` + `video_render_engine` `humo_1.7B` ->
   `humo_14B_169` (and resolve the visualizer drift per the open question).
2. Apply to the workflow via `nodes/_otr_workflow_apply.py` (NOT raw node-id patching) so
   `OTR_VideoDirector.other_beats_video_model` + `OTR_VideoRenderBatch.engine` =
   `humo_14B_169`; re-pin the capability-profile fixtures so the 5 drift-fails go green
   (or are correctly re-baselined).
3. Validate: `OTR_WorkflowValidator` + JSON round-trip + link/widget audit + the widget-count
   vs INPUT_TYPES check; then `test_capability_profiles.py` + `test_workflow_apply.py` + full
   suite + Bug Bible + B7.
4. VRAM safety acceptance: confirm HuMo stays single-resident (AS-3 lease); bound beat length
   to the tested-safe envelope (<=~49-81f @832x480); keep the post-decode reclaim. DECIDE
   whether to also port the pre-sampler two-stage encoder evict into `eng_humo.render_clip`
   (Step A says it buys only ~217 MB -- likely NOT worth the eng_humo change; confirm).
5. Live acceptance: a real episode render on the 5080 with the 14B, confirm OBS publish +
   no OOM at a representative + a max-safe beat length; operator eyeball the episode.

## HARD constraints
Single resident <=14.5 GB is the INVARIANT the 14B technically violates (~15.9 GB) -- the
operator has ACCEPTED this thin-headroom risk for the quality; the plan must make the
acceptance explicit + bound it (single-resident + beat-length cap), not silently break the
invariant. UTF-8 no BOM; SFW; commit per green chunk to v2.0-alpha; prod/main + tags GATED;
workflow + code in the SAME change.
