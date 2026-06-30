# OTR COVERAGE-SOAK FIX SPRINT -- WIRE-READY PLAN (r3-hardened)

Improved after kibitz r3 (Codex/gpt-5.5 wiring round + Claude grounding vs the live JSON). Codex-only
panel. All r3 corrections verified.

## r3 JUDGMENT
- ACCEPTED (verified): S-F must submit a PRUNED API prompt (render-tail nodes only) + patch node-92
  literals -- severing links alone fails because ComfyUI executes EVERY OUTPUT_NODE; allow_auto_fallback
  = DEPRECATE-IN-PLACE (delete shifts wv14..18; otr_api.py:608-617 fails on length drift); labels MUST be
  `engine_id (...)` (else `_engine_id_from_pick` breaks); recipe-stamp extends the existing
  `meta.render_engines` versioned payload; S-B acceptance must pin `OTR_LTX_AV_RENDER_CANVAS`; S-A must
  decide should_loop BEFORE the underrun warning.

## INVARIANTS (unchanged)
Workflow JSON source of truth (one validated migration per change; positional widgets); single heavy
<=14.5 GB; local/offline; seed-keyed; UTF-8 no BOM; SFW; master audio byte-identical (FROZEN); NO
FALLBACKS (already enforced at render time); suite+Bug Bible+B7 every change; commit+push per green chunk.

## CORRECTION: HuMo "mush" is NOT cfg -- it is the clip underrun (S-A). Motion is audio-driven.

## ORDER
S-F (bake from a CLEAN no-underrun reference) -> S-A -> S-B -> S-D -> S-E core -> S-C (split) -> BUG-411
(look-QA verify). visualizer_rainbow deferred.

---

## S-F  VISUAL SMOKE FIXTURE  [ACCELERATOR]
MECHANISM (r3-resolved): submit a PRUNED ComfyUI API prompt containing ONLY the render-tail nodes
(node 92 `OTR_VideoRenderBatch` + any required validator, e.g. node 63), NOT the full workflow --
because ComfyUI executes EVERY OUTPUT_NODE in a submitted prompt (SceneSequencer / MasterAudioMux /
SceneAwareScopes / PostUpscaleProcgenBlend would otherwise re-pull the writer+audio graph;
`scene_sequencer.py:1121`, `otr_master_audio_mux.py:246`, `otr_scene_aware_scopes.py:400`,
`otr_post_upscale_procgen_blend.py:887`). DIRECTLY PATCH node-92's converted API-prompt inputs to baked
literals -- `patched_ledger_json`, `master_audio_path`, `image_done` (forceInput defaults at
`otr_video_render_batch.py:77-99`; empty/invalid fixture JSON fails at :173-181) -- do NOT add constant
producer nodes (`otr_api.py` converts widgets/links, no constant-node interface). Build on
`otr_coverage_sweep.py` + `otr_api.py`.
BAKE: one good 30-word episode's master audio + ledger, from a CLEAN reference (no underrun).
ACCEPTANCE: the ComfyUI `/history` executed-node list contains ONLY the render-tail node ids (writer
node 1 + audio nodes ABSENT); audio byte-identical to the baked master; time = render tail; stamp a
fixture HASH per leg.

## S-A  CLIP-FILL + LEGIBILITY FLOOR  [HIGH]
ROOT CAUSE: clips underrun; the composite holds the last frame (`tpad=clone`). `_warn_clip_underrun`
(`otr_silent_composite.py:243`) already detects. CAP: `_HUMO_MAX_FRAMES=177` for base humo/1.7B; only
`humo_14B_169` caps at 49 -> the fill is SIZE-AGNOSTIC (`frame_count < target`).
FIX (concrete): in `plan_timeline_segments` compute `should_loop` (real clip row: `exists && path &&
frame_count>0 && frame_count < target`) BEFORE `_warn_clip_underrun`, and pass that so the LOUD warning
does NOT keep firing once fill is on (`otr_silent_composite.py:243-257, 325-339`). Set `loop=True` for
those rows (reuse the credits-tail loop path); keep `tpad`/`loop=False` for floor/credits/non-loop
sources (`:360-368, 395-411, 645-649`).
LEGIBILITY GUARD (STAGED): FIRST commit = assert DELIVERED `frame_count == target` in the manifest +
`freezedetect` on the SILENT video only; add manifest fields `freeze_score`/`quality_status`/
`fail_reason` (`build_clip_manifest`, `render_driver.py:2000-2015`) + thresholds + exact hard-fail
location. DEFER sharpness-ratio gating to a later commit.

## S-B  ltx_audio_in VRAM FIT  [HIGH]
ROOT CAUSE: regression `7bbce1d8`+`fd9edc28` (~15.9 GB, `eng_ltx_av.py:687`); last-good `c4d7815b`
@512x288=13688 MB. FIX: observability first, then re-fit via recipe/quant/offload.
WIRING: canvas is a `render_driver` override AND env-overridable via `OTR_LTX_AV_RENDER_CANVAS`
(`render_driver.py:1171-1179`), NOT a node-87 widget. ACCEPTANCE must explicitly SET + RECORD
`OTR_LTX_AV_RENDER_CANVAS=512x288` (a stale env invalidates the VRAM run). All 3 slots; NVML <=14.5 GB;
audio byte-identical. Replace the stale `13688` comment with a bakeoff-manifest link.

## S-C  AUDIO-IN CONDITIONING  [split]
C1 = `audio_motion_profile` extraction + ledger stamp (schema field, producer node, IS_CHANGED/cache
key, PROVE conditioning WAVs never replace the master). C2 = per-engine consumers + HuMo phrase-chunking
(also attacks the S-A root). Probe-gated HQ tiers last.

## S-D  gemma normalize_length WRAPPER-KEY DRIFT  [MED]
FIX: add `@model_validator(mode="before")` on `RadioEditPlan` (`_otr_radio_editor.py:316-323`, NO
before-validator today) that unwraps EXACTLY `{"RadioEditPlan": {...}}` and REJECTS ambiguous/multi-key
wrappers -- NOT the shared tolerant core. Add the multi-key-wrapper regression in the SAME commit.

## S-E  NO-FALLBACKS CLEANUP + ENGINE-MENU + UX  [HIGH]
Runtime no-fallback ALREADY enforced (`render_shot` raises LOUD, `render_driver.py:1526-1553`).
- `allow_auto_fallback` = DEPRECATE-IN-PLACE (NOT delete this sprint): keep the node-87 input/widget
  (wv13) + `direct()` signature, force it false in the emitted policy, relabel "(deprecated)". Deletion
  would shift wv14..18 and `otr_api.py:608-617` hard-fails on widget-length drift; do a clean delete only
  as a SEPARATE JSON rebaseline. (`otr_video_director.py:216,278-283,342`.)
- Fallback constants: `make_fallback_of` is STILL CALLED (run_real_episode `render_driver.py:1737`,
  render_single `:2217`); constants referenced (`:46-58,146,157-159,2282`; `eng_character_3d.py:55,257,
  326`) + asserted in tests. Migrate call sites + imports + tests in ONE commit (or keep no-op dead
  stubs until tests migrate).
- ENGINE RETIREMENT (low JSON risk -- verified): current node-87 saved values are
  visualizer/flux_gen1/humo_14B_169 + defaults -- NONE references `still_motion`/`station_card`/
  `abstract`, so unregister won't orphan a widget value; dropdown options regenerate from the registry.
  ACTION: verify no `capability_profiles`/role-default NAMES a retired engine, then unregister
  (`cheap_families.py:165-190`, `registry.py:127-133`) + update tests.
- DROPDOWN LABELS: MUST keep the form `engine_id (display metadata...)` -- `_engine_id_from_pick` parses
  the text BEFORE the first `" ("` (`otr_video_director.py:61-87`, combo at :105-125); a free-form label
  resolves to a non-engine and fails validation. Add a registry display-metadata field; render labels as
  `humo_1.7B (HuMo 1.7B, portrait, LOW-VRAM ~3.3 GB)`, `humo (HuMo 14B, portrait, HIGH-VRAM ~15.9 GB)`,
  etc. KEEP BOTH HuMos.
- RECIPE-STAMP: extend the EXISTING versioned `meta.render_engines` payload (today: histogram /
  video_revision / by_role / vram_peak; `otr_video_render_batch.py:26-49`) -- preserve those keys, add a
  `per_clip`/`by_engine` recipe block (`delivered_engine` + `recipe/quant`). PREREQ: engines must RETURN
  `recipe/unet/lora/quant/canvas/audio_source/phase` in the raw/canonical clip (`eng_ltx_av` returns only
  out_path/frame_count/vram_peak_mb; `_clip_from_raw` keeps only vram_peak_mb,
  `eng_ltx_av.py:621-627,691-694,776-788`) -> thread into `build_clip_manifest` -> the payload. `meta` is
  per-key merged by `_merge_with_disk` (durable); do NOT touch `TOP_PRESERVE`.
- ANNOUNCER + MUSIC = always a radio-themed still (broadcast-studio vocab already at
  `otr_meta_brief_image_prompt.py:127`); never black/abstract/fallback.
- ADD `visualizer_rainbow` -- DEFERRED (own `/roundtable` for the GLSL shader stack first).

## BUG-411  [look-QA verify -- DONE except one check]
FluxGuidance @3.5 (`flux_gen1.py:88-92,130-135`), cinematic grade tail
(`otr_meta_brief_image_prompt.py:535`), radio distress tail (:805), portrait STYLE_ANCHOR (:92) ALL
present. VERIFY only the bookend seed 4242; implement only if missing. No JSON node.

## OPEN FOR r4 (convergence)
Confirm no NEW must-fix introduced by the r3 resolutions: (1) the pruned-prompt S-F path renders a
complete episode (all 6 beats present, no missing-dependency error); (2) the deprecate-in-place
allow_auto_fallback keeps `otr_api.py` conversion green; (3) the `engine_id (...)` label format passes
`_engine_id_from_pick` for every engine; (4) the should_loop-before-warn change keeps
`test_audio_byte_identical` + the composite tests green.
