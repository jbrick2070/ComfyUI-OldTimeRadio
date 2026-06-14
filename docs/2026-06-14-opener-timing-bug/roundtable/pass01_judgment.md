# Roundtable pass01 judgment — opener-timing bug (BUG-403/404)

Panel: GPT-5.5 (`openai/gpt-5.5-20260423`), Gemini 3.1 Pro (`google/gemini-3.1-pro-preview`),
DeepSeek-v4-pro. Spend ~$0.33. Claude = sole judge/grounder. **CONVERGED in one pass** (all three
agreed on the architecture; grounding resolved the single disagreement + caught one misread).

## CONFIRMED (grounded against the real code) — folded into the fix
- **Fix home = `EpisodeAssembler`, NOT `SceneSequencer`** (all 3). GROUNDED: `scene_sequencer.py`
  EpisodeAssembler prepends `opening_theme_audio` (L1059-1060) and computes the master-timeline shift
  `_shift_s` (BUG-LOCAL-106, L1239-1276: `_shift_samples = sum(segments before scene) - xfade*scene_idx`).
  SceneSequencer writes `start_s` in **scene-audio space** (L923 `"start_s_space": "scene_audio"`) — so
  stamping there would be in the wrong timeline. => **REJECT pass00 option A (SceneSequencer).**
- **Opening lead-in duration = `_shift_s`** (the master-space offset EpisodeAssembler already computes),
  NOT a hardcoded 9.5 and NOT blindly `first_dialogue.start_s` (GPT #2: with a 10.0s theme + 500ms
  crossfade, the segment is 10.0s but the scene shift is 9.5s; the non-overlapping visual lead-in is
  `_shift_s` = 9.5). GROUNDED: the xfade term is real (L1272 `_shift_samples -= xfade_samples*scene_idx`).
- **A manifest CLIP ROW is also required** (GPT #4, DeepSeek #2). GROUNDED: `plan_timeline_segments`
  reads `manifest["clips"]` rows with `target_frame_count>0` only — it does NOT read stills on disk. The
  existing `b000_music_open` still is inert until a clip-manifest row exists. The manifest builder
  (`OTR_VideoRenderBatch` / render_driver `build_clip_manifest`) must emit a b000 row: `start_s=0`,
  `target_frame_count=round(_shift_s*fps)`, path/shot_id -> the b000 still.
- **Title-card fallback is a SEPARATE defect (BUG-404)** (all 3). My fallback should set the window to
  `[0, round(_shift_s*fps))` from the first dialogue and draw on the floor head-gap, but the opener is
  black — needs its own diagnosis (does the fallback fire? is the head-gap floor slice shown? is the
  floor's first ~238 frames black?). Not auto-fixed by the line/clip repair.
- **Reject Option C (composite infers untimed leading music)** as a shim (all 3). The composite sees only
  `base_video_path` + `clip_manifest_json`; it cannot know the resampled/crossfaded opening duration.
- **Derive everything from real values** (all 3): `_shift_s` from the prepended theme, never hardcode.

## MISREAD (caught by grounding — do NOT build on it)
- GPT/DeepSeek invoked an "existing music-mirror (BUG-LOCAL-130) that stamps `ledger.music[]` ->
  `lines[]` and SKIPPED because `ledger.music[]` was empty." GROUNDED: there is **no such music mirror**.
  The SFX mirror was DELETED as a permanent no-op (L945-950); music lines are explicit PASSTHROUGH and
  never timed (L706-713); EpisodeAssembler only SHIFTS existing lines (BUG-106), it never CREATES a
  music_open line. => The accurate root cause is **"EpisodeAssembler synthesizes no opening-music line at
  all,"** not "a mirror skipped." Gemini's instinct (synthesize the row in EpisodeAssembler) is right;
  the BUG-130-mirror premise is wrong.

## DISAGREEMENT RESOLVED by grounding
- GPT #5 + Gemini #2 said an untimed `b000` row forces the WHOLE composite into SEQUENTIAL mode
  (`all(start_s is not None)`), overwriting the floor title frames. DeepSeek corrected: if the row is
  **absent entirely** (this episode: ledger had ZERO music lines), POSITIONED mode HOLDS (all 5 dialogue
  rows are timed) and the head-gap is floor/black — not a broken mode. GROUNDED: `positioned = ... and
  all(r.get("start_s") is not None for r in rows)`. => **DeepSeek is right for this episode** (no b000
  row -> positioned -> head-gap IS floor -> the black + missing card means the FLOOR isn't drawing the
  title card). The sequential-mode trap is a REAL latent hazard but only bites if an untimed b000 row is
  ever added — keep the `all()`->`any()`+filter as a defensive SHOULD-FIX, not the root.

## STILL-OPEN verify-at-build (for the coder)
- Confirm whether this episode's clip MANIFEST had a b000 row at all (the audio-stage ledger had
  `video.shots == []`). If absent -> positioned mode held -> BUG-404 is a pure floor-title-card render
  miss. If an untimed b000 row was present -> the sequential trap also bit (do the `any()` fix too).
- BUG-404: instrument `_resolve_title_timing` output + the first 5 `plan_timeline_segments` segments
  (`order/source/shot_id/src_start_frame/n_frames`) on the failing ledger to localize "black because
  floor missing" vs "floor present but card not drawn" vs "b000 clip hides floor."
- Audit `_envelope_intro_end` — may return 0 for faint openings, disabling the fallback (DeepSeek).

## Convergence
One grounded pass; the panel agreed on the architecture, grounding flipped the fix home off SceneSequencer
onto EpisodeAssembler, corrected the "music-mirror" premise, and resolved the sequential-mode dispute. No
new material direction would come from a 2nd paid pass — STOP.
