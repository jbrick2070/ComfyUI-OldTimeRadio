VERDICT: no. The plan has the right root-cause direction, but the credit surface is split between early node-12 HUD and late terminal rendering without a coherent single viewer experience.

MUST-FIX BEFORE BUILD:
1. [Core problem / S3 / Proposed viewer roll] The plan says keep story credits early in node 12 and move engine credits late, but the proposed viewer roll is written as one unified roll. Current node 12 already builds an on-screen dossier and render-engine block, best-efforting missing fields to “(not recorded)” (nodes/video_engine.py:1068, nodes/video_engine.py:1073, nodes/video_engine.py:1078, nodes/video_engine.py:1140). Fix: define exactly one viewer-facing credits architecture. Preferred: make the full viewer roll late and strip node 12 to non-receipt story/floor behavior; otherwise explicitly specify two sequential surfaces and their timing.

2. [S1] The plan treats `meta.voice_cast_decision[char_id].accepted_id` as the real final cast voice, but CastLock may reject it and assign a deterministic alternative when the engine does not match, validation fails, or a collision occurs (nodes/cast_lock.py:565, nodes/cast_lock.py:570, nodes/cast_lock.py:574, nodes/cast_lock.py:596). Node 12 receives frozen script JSON from node 62, not CastLock output (workflows/otr_scifi_16gb_full.json:1, links 16 and 234-237). Fix: either render actual “CAST & VOICES” late from the CastLock-stamped singleton, or label early S1 output as planned voice fit rather than final credits.

3. [S3] Extending `OTR_PostUpscaleProcgenBlend` for late credits conflicts with its existing source-copy fallback: on ffmpeg blend failure it logs a warning and copies the source so the pipeline still produces a deliverable (nodes/otr_post_upscale_procgen_blend.py:1036, nodes/otr_post_upscale_procgen_blend.py:1039, nodes/otr_post_upscale_procgen_blend.py:1045). That would silently drop late credits. Fix: when credits are active, missing manifest/ledger or render failure must raise before mux; no source-copy fallback on the credits path.

4. [S1 / Proposed viewer roll] Music engine provenance has no durable or wired path into node 12. `OTR_StableAudioTheme` returns a `done` string and formats `music:done:engine=...` (nodes/stable_audio_theme.py:67, nodes/stable_audio_theme.py:173), but workflow node 83’s `done` output is not linked (workflows/otr_scifi_16gb_full.json:1). Fix: stamp music engine into ledger meta or wire it to the late credits node; do not rely on reading the workflow widget as the source of truth.

5. [S2 / Cleanbreak orchestration] “Mirror the proven pattern” conflicts with “no singleton missing, skip.” The video render stamp writes `led.data["meta"]["render_engines"]` and saves (nodes/otr_video_render_batch.py:71, nodes/otr_video_render_batch.py:74), while the image dispatcher only mutates the returned ledger JSON today (nodes/otr_image_gen_dispatcher.py:658, nodes/otr_image_gen_dispatcher.py:673, nodes/otr_image_gen_dispatcher.py:805). Fix: define the production persistence contract explicitly: required singleton save with loud failure in production, with a deliberate test-mode injection path if needed.

SHOULD-FIX:
1. [S3] Pick the terminal shape now. “Extend node 93 OR add a small new node” leaves the central architecture unresolved. Node 93 is currently after caption burn/scopes and before mux (workflows/otr_scifi_16gb_full.json:1), and MasterAudioMux consumes node 93 output (nodes/otr_master_audio_mux.py:252). Fix: choose the exact insertion point and ownership of output path, ledger final-path stamping, captions, bars, and mux input.

2. [S0 / S3] Font scale is scoped only to node 12, but the plan moves receipt credits late. Node 12 clamps HUD length to 20-90 seconds (nodes/video_engine.py:1352, nodes/video_engine.py:1356) and MasterAudioMux enforces a 45s silent-tail budget (nodes/otr_master_audio_mux.py:149, nodes/otr_master_audio_mux.py:150). Fix: define one shared duration/readability budget for early and late credits before changing font/speed.

3. [Validation gate] “grep the treatment / view frames” is too weak for the stated goal: late credits may not be in the treatment at all once S3 removes the early treatment merge. Fix: add a concrete frame-level verification target for the final mux input or final OBS copy, not only the sidecar treatment.

OPTIONAL / NICE-TO-HAVE:
- [S4] Version/footer and telemetry relabel are sensible polish, but they should not drive the structural design.
- [Proposed viewer roll] Date/GPU/git SHA are fine only if each has a declared ledger/source path.

CUT THESE (scope / over-engineering):
1. [S4] Cut the extended `OTR_CREDITS_DEBUG` card from the first build. Recipe/quant/LoRA/canvas/VRAM, degradation trail, OpenRouter cost, timings, SHAs, and story grades are forensic sidecar material; they do not serve the core viewer-credit fix and expand the late renderer’s surface area.

2. [S0] Defer the +50% font campaign until after the data path is correct. It is visual tuning, not credits enrichment, and it couples directly to the 90s HUD clamp and 45s mux guard (nodes/video_engine.py:1356, nodes/otr_master_audio_mux.py:149).

3. [S3] Cut candidate B unless the plan also moves episode finalization. The document already identifies node 12 finalization/rename hazards; keeping B in the active build plan invites a known bad path.