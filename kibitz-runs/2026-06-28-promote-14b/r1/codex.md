VERDICT: no. The plan asks “where does the 14B go?” but the coding slices already force it into the shared other-beats slot, which also drives face-less roles.

MUST-FIX BEFORE BUILD:
1. [OPEN QUESTION + Coding slices 1-2] The plan is internally unresolved: it says not to guess whether 14B belongs on audio-driven face beats or general other-beats, then mandates `other_beats_visual` / `OTR_VideoDirector.other_beats_video_model` = `humo_14B_169`. That slot feeds `character_video`, `scene_broll`, and `background_abstract` in `C:\Users\jeffr\Documents\ComfyUI\custom_nodes\ComfyUI-OldTimeRadio\nodes\otr_shot_lock.py:709-714`; HuMo requires `audio_ref` + `init_image` in `...\nodes\_otr_video_engines\eng_humo.py:81-89`; `scene_broll` lacks `audio_ref` and `background_abstract` has only text in `...\nodes\_otr_shared\role_compat.py:55-72`. Concrete fix: pick one route before build. If 14B is for talking-face beats, split/target `character_video` separately instead of the shared other-beats slot. If it is for all other-beats, define explicit behavior for scene-broll/background failures.

2. [Grounded facts + Coding slices 2] The plan treats node 92 as part of the production promotion while also saying node 92 is ignored in episode mode. Code confirms episode mode bypasses the `engine` widget and renders from the ShotLock ledger in `...\nodes\otr_video_render_batch.py:127-134`; the saved workflow has node 92 `engine=humo_1.7B` but node 87 other-beats `visualizer (16:9)` in `...\workflows\otr_scifi_16gb_full.json`. Concrete fix: state node 92 is parity/smoke only, and make live acceptance prove the ShotLock ledger rows / render manifest used `humo_14B_169`, not merely that node 92 changed.

3. [Coding slices 4 + HARD constraints] The beat-length cap is only prose. Current HuMo clamps to `_HUMO_MAX_FRAMES = 177` in `...\nodes\_otr_video_engines\eng_humo.py:54` and passes that into frame planning at `...\nodes\_otr_video_engines\eng_humo.py:341`; the verdict’s tested envelope cites `<=49f` / `49-81f` in `...\docs\2026-06-27-humo-bakeoff\FINAL_VERDICT.md`. Concrete fix: choose the enforcement point and over-length policy before build: render-driver clamp, ShotLock budget split, fallback, or hold-frame extension. Add acceptance that a max-safe beat cannot exceed the chosen cap.

4. [Coding slices 4] The plan contradicts its own source of authority on two-stage evict. `FINAL_VERDICT.md` says the promotion mitigation is to “keep the two-stage encoder evict”; the plan says decide later and leans “likely NOT worth” it. The same verdict records single 15996 MB vs two-stage 15779 MB in `...\docs\2026-06-27-humo-bakeoff\FINAL_VERDICT.md`. Concrete fix: either make two-stage evict part of the promotion, or remove it from the safety story and rebase acceptance on no-evict runs.

SHOULD-FIX:
1. [Grounded facts] “5 pre-existing suite fails” is not grounded in the reviewed source, only asserted in `...\kibitz-runs\2026-06-28-promote-14b\r1\input.md`. Concrete fix: list exact failing test names or attach the test log; otherwise mark as `verify: current failing tests`.

2. [Coding slices 2] “re-pin fixtures” underspecifies whether tests are being updated to new truth or weakened to pass. The identity/drift tests are real invariants in `...\tests\test_capability_profiles.py:176-205` and `...\tests\test_workflow_apply.py:111-117`. Concrete fix: state the new invariant: profile and saved workflow must match exactly after the chosen routing decision.

3. [HARD constraints] The plan assumes HuMo is enabled and installed. Code gates HuMo behind `OTR_ENABLE_HUMO` / model presence in `...\nodes\_otr_video_engines\eng_humo.py:13-14` and raises unusable errors at `...\nodes\_otr_video_engines\eng_humo.py:150-156`. Concrete fix: add a preflight step for flag + model files before live acceptance.

OPTIONAL / NICE-TO-HAVE:
1. Add a post-render report check for `meta.render_engines.engine_histogram`; stamping exists in `...\nodes\otr_video_render_batch.py:26-44`.

CUT THESE (scope / over-engineering):
1. [Coding slices 4] Cut “DECIDE whether to port pre-sampler evict” as an open-ended build task. It is either required mitigation or not; leaving it as a mid-build decision is scope creep.

2. [Coding slices 2] Cut any raw “re-pin fixtures” language. Replace with one explicit invariant update after routing is chosen.