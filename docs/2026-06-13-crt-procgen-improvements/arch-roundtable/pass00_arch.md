# v2 SCENE-AWARE REACTIVE OVERLAY -- architecture to harden (round robin)

GOAL: draw the audio-reactive gutter SCOPES scene-aware (in the REAL per-beat gutters --
portrait beats have gutters, landscape b-roll does not) WITHOUT relocating the procgen
floor. This is the proper fix for the landscape-gutter open decision in §4C.

## Grounded pipeline (the real nodes + IO)
1. `OTR_SignalLostVideo` (`render_video` -> `_CRTRenderer`): renders the procgen FLOOR --
   CRT texture (scanlines/vignette/grid/noise) + the TITLE CARD + (v1) the scopes -- at
   `resolution` (default 1920x1080) @ `fps` (default **24**), from the FULL master audio.
   Inputs: `audio, script_json(ledger), news_used, fps, resolution, episode_title,
   closing_audio`. NO manifest, NO per-beat engine plan. The floor is ALSO the gap/credits
   fill and the green-blend source.
2. `OTR_SilentComposite`: assembles real clips into a 1472x832 @ **25fps** SILENT timeline;
   gaps filled from the floor. Holds the **clip manifest** (`clips[]` with per-beat
   `engine_id`, `shot_id`, `start_s`, `target_frame_count`, `exists`, `path`).
3. `RTXUpscale`: 1472x832 -> 1920x1080.
4. `OTR_PostUpscaleProcgenBlend`: GREEN-ONLY `screen` blend of the floor procgen over the
   upscaled composite; audio `-c:a copy`. Inputs today: `source_mp4, procgen_mp4, blend
   params, captions`. NO manifest.
5. `OTR_MasterAudioMux`: muxes the frozen master audio.

Aspect: HuMo character beats are 480x832 -> pillarboxed -> ~647px gutters each side at
1920; LTX/Wan b-roll beats are landscape full-frame -> NO gutters. The per-beat
portrait-vs-landscape is derivable from `engine_id` (registry aspect) and/or the clip's
native dims -- known ONLY at the manifest (step 2) and later.

## Proposed v2 design (the thing to attack)
- **Split ONLY the scopes** out of the floor. In v2 the floor draws CRT texture + the
  title card (time-based) but NOT the scopes. The scopes move to a late ADDITIVE pass.
- **New node `OTR_SceneAwareScopes`** placed AFTER the upscale/blend (post-upscale, so the
  synthetic scope geometry stays crisp -- same reason procgen is post-upscale). Inputs:
  the upscaled+blended `source_mp4`, the `clip_manifest_json`, the master `audio` (or its
  analysis), and the scope params. Output: an mp4 with the scene-aware scopes composited
  green-only over the input; audio `-c:a copy` (spine untouched).
- **Per frame `fi`:** map `fi -> t = fi/fps_delivery -> the active beat` whose
  `[start_s, start_s+dur_s)` contains `t`; read its `engine_id -> aspect`. Portrait ->
  draw the two scopes in the real gutters (the same `_draw_fft_scope`/`_draw_scope`
  module). Landscape -> suppress the scopes (or draw an ultra-dim edge variant). Gap beats
  (no clip) -> treat as the floor (full-frame) -> suppress.
- **Audio analysis** is RECOMPUTED in the late node from the master audio via the same
  `_analyze_audio`, at the DELIVERY fps (25, not the floor's 24) -- deterministic, so it
  matches frame-for-frame at the delivery rate.
- **The scope drawing module is stage-agnostic:** v1 calls it from the floor
  (beat-agnostic, 24fps), v2 from the late node (scene-aware, 25fps). Same code; only the
  caller + the gutter-geometry source (computed vs per-beat-manifest) + the fps differ.
- Determinism (stable-hash RNG), Pillow-only/CPU, no new model, green-only output.

## Open architecture questions for the panel
1. NEW node vs EXTEND `OTR_PostUpscaleProcgenBlend` (which already does the green
   composite). One composite pass (extend, but the blend is pure-ffmpeg today -> becomes a
   PIL+ffmpeg hybrid) vs two passes (new node, simpler but an extra encode).
2. fps: the floor is 24, the composite/delivery is 25. v2 scopes run at the DELIVERY fps;
   the audio re-analysis must be at 25fps. Confirm no 24/25 drift in the beat mapping.
3. Beat-boundary flicker: the gutter-state flips at beat seams (portrait->landscape) ->
   the scopes pop in/out. Need a crossfade / hold at seams?
4. engine_id -> aspect: is the registry aspect reliable per engine, or must we probe each
   clip's native dims from the manifest `path`? What about `still_*`/`visualizer` engines?
5. Two green layers: the floor green blend (CRT+title) + the scope green overlay. Order +
   interaction (both `screen` green-only -> additive, should be safe). Confirm.
6. The floor still renders scopes in v1; switching to v2 means the floor STOPS drawing
   them -- a flag/branch. Keep both paths or hard-switch?
7. Master audio availability at the late node: the frozen master WAV is in the episode
   folder (the composite/mux path knows it). Confirm the late node can resolve it for the
   re-analysis.
8. Cost: a full-episode per-frame PIL draw + composite at 1920x1080@25 (same cost as the
   floor does today, just relocated). Acceptable?

## INVARIANTS (must hold)
No audio-spine touch (`-c:a copy`); green-only; deterministic per seed; Pillow-only/CPU;
no new model; the procgen floor's gap-fill/credits/blend roles UNCHANGED; the title card
+ CRT stay in the floor.
