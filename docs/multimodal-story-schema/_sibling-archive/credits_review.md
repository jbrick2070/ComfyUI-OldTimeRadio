# Credits Review — end-roll model listing, added telemetry, font size

Date: 2026-07-03
Scope: `ComfyUI-OTR-UpstreamStoryLab` (this staging repo only). Read-only code
audit. No files changed except this one.

Request being answered:
1. The image and video models were supposed to be added to the credits that
   roll at the end. Confirm whether that was done.
2. Look for more content / telemetry we can add to the credits.
3. Consider bumping the credits font size by ~50%.

---

## TL;DR

- **It was not done, and it cannot be done from this repo alone.** Two blocking
  facts:
  - **The end-credits roll renderer does not exist in this repo.** There is no
    end-card / credits-roll / info-card / drawtext / font code anywhere here.
    That surface lives in the parent `ComfyUI-OldTimeRadio` repo (this is the
    staging/transplant repo). See "Where the credits actually live" below.
  - **The image (FLUX) and video (LTX / HuMo) model identities are never
    captured upstream in the first place.** Only the two writing LLMs
    (`creative_model`, `technical_model`) plus OpenRouter provenance get stamped
    into the run metadata. So even the parent renderer currently has nothing to
    print for image/video — the data never reaches it.
- Net: this is not a "add two lines to the credits template" fix. The image and
  video model names have to be *captured* first (prepare the seam here), then
  *rendered* (in the parent repo).
- **Font size:** there is no text-drawing code in this repo, so there is no
  50%-bump target here. The font size lives with the renderer in the parent
  repo.

---

## Where the credits actually live (not here)

The render chain in `production_mirror/workflows/otr_scifi_16gb_full.json` is
`OTR_VideoDirector -> ... -> OTR_CaptionBurn -> OTR_PostUpscaleProcgenBlend`.
None of these node classes have a Python implementation in this repo — they are
referenced but defined in the parent `ComfyUI-OldTimeRadio` repo (confirmed by
`CLAUDE.md:50` and `docs/2026-07-01-code-ready-kibitz/UPSTREAM_STORY_LAB_CODE_READY_BRIEF.md:7-17`).

`OTR_CaptionBurn` only burns SDH dialogue subtitles; it is not the credits card.
A viewer-facing "video info card" surface is named as a downstream consumer in a
comment at `production_mirror/nodes/OTR_LedgerScriptWriter.py:295`, but the code
that draws it is not in this repo.

Implication: the actual credits-roll edit (adding lines, changing font) has to
happen in `ComfyUI-OldTimeRadio`. This repo's job is to make sure the model
names it needs are present in the run metadata it reads.

---

## What model provenance is captured today

All of it is LLM-only. Stamped in `production_mirror/nodes/OTR_LedgerScriptWriter.py`:

- `meta["creative_writing_model"]` and `meta["technical_model"]` (~lines 5303-5304)
- `meta["creative_model"]` legacy alias + `meta["creative_prompt_profile"]` (~lines 5506-5515)
- OpenRouter provenance via `openrouter_meta_for()` / `openrouter_run_meta()`
  (~lines 5517-5535). Note: the backing module `_otr_openrouter_backend` is a
  parent-repo dependency, not present here.

What is **missing** (never stamped anywhere):

- Image model: **FLUX** — appears only as a workflow widget value
  (`flux_gen1`) and a widget-name constant in
  `production_mirror/nodes/_otr_workflow_apply.py:145-147`. Never written to meta.
- Video models: **LTX / HuMo / character_video / b-roll / lipsync** —
  `humo_14B_169` / `humo_1.7B` are workflow widget values only. Never written
  to meta.
- Audio / TTS: **Bark / Kokoro** — workflow widget values only.

There is a second, disconnected provenance structure: `build_clip_manifest()` in
`production_mirror/nodes/_otr_video_engines/render_driver.py:2115-2231`
aggregates per-episode `engine_id` / `engine_histogram` (which video engines
actually ran, and how many clips each). This is exactly the video-model data the
credits want — but it travels on a separate `clip_manifest_json` socket and is
**never merged into `led.data["meta"]`**. The LLM-provenance path (ledger meta)
and the image/video-engine path (clip manifest) do not meet today.

---

## Fix plan (prepare the seam here, render in parent)

### Step 1 — capture image + video models into meta (this repo)

- `render_driver.py`, `build_clip_manifest()` (lines 2115-2231): add a
  `manifest["models_used"]` rollup = image model + sorted `engine_histogram`
  keys (the video engines that actually rendered). The engine IDs are already
  aggregated here, so this is a small additive change.
- Add a merge step so this manifest is folded back into `led.data["meta"]`
  (e.g. `meta["image_model"]`, `meta["video_models"]`, `meta["audio_models"]`).
  That merge step does not exist yet and is the real missing wire.
- Caveat: image/video engine choices are made in the render nodes, which run
  *after* `OTR_LedgerScriptWriter`. So do not try to stamp them inside the
  writer's ~line 5498-5535 block (that block only knows the LLMs). The clip
  manifest is the correct, already-post-render place to source them.

### Step 2 — render them in the credits (parent repo, `ComfyUI-OldTimeRadio`)

Locate the info-card / credits-roll renderer there and have it read the new
`meta` keys. This is where the actual "Image model: FLUX / Video model: HuMo,
LTX" lines get added. Out of scope for this repo, but this is where the user's
original ask ultimately completes.

---

## More content / telemetry available for the credits

All already captured; a credits card could pull any of these once the meta seam
is wired. Grouped by kind.

Story facts (`OTR_LedgerScriptWriter.py`):
- `episode_title` + `title_source` (how the title was derived)
- `total_word_count` / `character_word_count` / `announcer_word_count`
- `est_minutes` (estimated runtime; currently a node output socket, not in
  meta — capture at graph-output time)
- `theme`, `style_descriptor`, `story_contract` (slug / label / ending_tag /
  sound_world)
- cast: `cast_voice_slots`, `cast_seed` + `cast_seed_source`,
  `visual_plan.characters` (name -> portrait prompt)

Model / engine provenance:
- LLMs: `creative_writing_model`, `technical_model`, `creative_prompt_profile`,
  `creative_repo_id` / `technical_repo_id`, OpenRouter provenance
- Video engines: `engine_histogram` (per-engine clip counts), `final_engine`
  per shot (post-fallback), `engine_id` / `family` / `role` per shot
- (image / video / audio model *names*: add per Step 1)

Render telemetry (`render_driver.py`):
- `elapsed_s` wall-clock, `vram_peak_mb` / `vram_ceiling_mb`
- `degradation_trail` / `runtime_fallback_decisions` (which fallbacks fired)
- `clip_count` / `n_beats` / `total_target_frames`, resolution + fps (`canvas`)
- `seed_bundle.request_seed` / `video_seed`, `audio_sha` / `master_audio_sha256`

QA / integrity:
- `story_quality.refine_loop` (winner grade, pass count, stop reason),
  `story_quality.best_of_n`, `consistency_status`
- freeze cascade: `freeze_verdict`, `freeze_timestamp` (ISO-8601),
  `cleanup_locked`, `gap_audit_pre` / `gap_audit_post`

Suggested tasteful credit content (not the whole dump): title, estimated
runtime, cast + voices, writing model(s), image model, video model(s), audio
model(s), seed, and a short "made with" line. Keep the deep telemetry
(vram, shas, fallback trails) for a debug card, not the viewer roll.

---

## Font size (+50%)

There is **no font / text-drawing code in this repo** — confirmed by exhaustive
grep (no PIL `ImageFont`, no ffmpeg `drawtext`, no `cv2.putText`; the only
ffmpeg call at `render_driver.py:361-369` is audio-only, `-vn`). So there is no
existing default here to multiply by 1.5.

The credits font size is a parameter of the renderer in `ComfyUI-OldTimeRadio`.
The 50% bump has to be made there, at whatever `font_size` / point-size the
drawtext or PIL compositing step uses. Recommendation: when that renderer is
touched, expose `font_size` as an explicit named parameter rather than a magic
number, so a "+50%" or accessibility bump is a one-line config change instead of
a hunt.

---

## Minor: stale "treatment" term

Per the note that "treatment" is no longer used: the word survives in one code
comment listing downstream consumer surfaces —
`production_mirror/nodes/OTR_LedgerScriptWriter.py:295` ("HUD overlay, FLUX
scene-prompt composition, treatment txt, video info card"). It is only a
comment, not live code, but if "treatment" is retired terminology it should be
cleaned up there (and swept in the parent repo) so the comments match current
usage. Everywhere else "treatment" appears is in docs, not code.

---

## Verification

- Confirmed no credits/end-card/font code in-repo via repeated grep (py/json/md)
  for credits, end card, info card, drawtext, ImageFont, font_size, overlay.
- Confirmed LLM-only meta stamping and absent FLUX/LTX/HuMo stamping by reading
  the stamping block in `OTR_LedgerScriptWriter.py` (~5498-5535) and the widget
  value refs in `_otr_workflow_apply.py`.
- Confirmed parent-repo ownership of the render nodes via `CLAUDE.md:50` and the
  code-ready brief.
- Confirmed `build_clip_manifest()` aggregates engines but does not merge to
  ledger meta (`render_driver.py:2115-2231`).
