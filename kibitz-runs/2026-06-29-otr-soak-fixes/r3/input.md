# OTR COVERAGE-SOAK FIX SPRINT -- WIRE-READY PLAN (r2-hardened)

Improved after kibitz r2 (Codex/gpt-5.5 coding-plan round + Claude grounding). All r2 corrections
verified against the real files. Codex-only panel (Antigravity out of credits).

## r2 JUDGMENT
- ACCEPTED (verified): recipe-stamp must extend `meta.render_engines` (already durable), NOT
  `TOP_PRESERVE`; the 49-frame cap is ONLY `humo_14B_169` (base/1.7B cap at 177) -> frame fill must be
  size-agnostic; `make_fallback_of` is STILL called (run_real_episode / render_single) so S-E migrates
  call sites + tests together; node 91 does NOT take `audio_done` (node 90 does); S-F non-execution is
  STRUCTURAL link-severing, not "cache invalidation" (no such API in otr_api.py); S-D unwrap belongs on
  the RadioEditPlan schema, not the shared core; engines must RETURN recipe fields before they can be
  stamped.
- STAGING: S-A first commit = frame-count + freezedetect ONLY; sharpness-ratio gating is a later commit.

## INVARIANTS (unchanged -- a fix that breaks one is rejected)
Workflow JSON is source of truth (change IN it, same commit; positional widgets append-only; re-validate);
single heavy <=14.5 GB; local/offline; seed-keyed determinism; UTF-8 no BOM; SFW; master audio
byte-identical (FROZEN spine; conditioning WAVs model-input only); NO FALLBACKS (already enforced at
render time); suite + Bug Bible + B7 every change; commit+push per green chunk to v2.0-alpha.

## CORRECTION (do NOT chase): HuMo "mush" is NOT a cfg regression
cfg defaults deliberate/measured; motion is AUDIO-driven; real cause = clip underrun (S-A).

## ORDER
S-F first (bake from a CLEAN no-underrun reference) -> S-A -> S-B -> S-D -> S-E core -> S-C (split) ->
BUG-411 (look-QA verify). visualizer_rainbow deferred (own roundtable).

---

## S-F  VISUAL SMOKE FIXTURE  [ACCELERATOR]
SEAM: `_otr_soak_phase0/1.py` DO NOT EXIST. Build on `scripts/otr_coverage_sweep.py` +
`scripts/_otr_soak_capstone.py` + `scripts/otr_api.py`.
NODE I/O (CORRECTED, grounded vs the JSON): node 91 = `script_json`, `image_policy_json`,
`image_prompts_json`, `gate_in`, `episode_id` (NOT audio_done); node 90 owns `audio_done`; node 92 =
`patched_ledger_json`, `master_audio_path`, `image_done`.
INJECTION = APPROACH A only (cut the fixture-loader node -- a test-only accelerator must not add a
production node): at the API layer, add/replace CONSTANT STRING producer nodes feeding the node 91/92
sockets with the baked fixture values, AND SEVER the upstream links to the writer (node 1) + audio nodes
so they are no longer DEPENDENCIES -> ComfyUI does not execute them. otr_api.py has widget-patch + queue
helpers (otr_api.py:541-656, 851-866) but NO cache-invalidation API -> non-execution must be STRUCTURAL
(severed links), not a cache call.
BAKE: one good 30-word episode's master audio + ledger (cast/brief/beats/durations/portrait hashes) from
a CLEAN reference (no underrun).
ACCEPTANCE: trace proves writer + audio nodes were NOT executed; audio byte-identical to the baked
master; time = render tail; stamp a fixture HASH on each leg report.

## S-A  CLIP-FILL + LEGIBILITY FLOOR  [HIGH]
ROOT CAUSE: clips underrun the beat; the composite holds the last frame. `otr_silent_composite.py:243`
already WARNS (detection done); the FILL is missing. CAP PRECISION (CORRECTED): `_HUMO_MAX_FRAMES=177`
caps base `humo`/`humo_1.7B`; only `humo_14B_169` caps at 49 (`safe_render_frames`). So the fix is
SIZE-AGNOSTIC: fill ANY `clip.frame_count < target_frame_count`.
FIX (concrete): in `plan_timeline_segments` set `loop=True` ONLY for real clip rows where
`exists && path && frame_count>0 && frame_count < target_frame_count`; keep `tpad`/`loop=False` for
non-loop / floor / credits sources (`otr_silent_composite.py:325-339, 360-368, 395-411, 645-649`). The
loop path already exists for the credits tail -- reuse it.
LEGIBILITY GUARD (STAGED): FIRST commit = assert DELIVERED `frame_count == target` in the manifest +
`freezedetect` on the SILENT video only (never the master). Add manifest fields `freeze_score`,
`quality_status`, `fail_reason` (build_clip_manifest, `render_driver.py:2000-2015`) + thresholds + the
exact HARD-FAIL location. DEFER sharpness-ratio gating to a later commit (less plumbing first).
NOTE: S-C HuMo phrase-chunking attacks the same root from the audio side -- panel r3: confirm composite-
fill is PRIMARY, chunking is the audio-quality follow-on.

## S-B  ltx_audio_in VRAM FIT  [HIGH]
ROOT CAUSE: regression `7bbce1d8`+`fd9edc28` (~15.9 GB, `eng_ltx_av.py:687`); last-good `c4d7815b`
@512x288=13688 MB. FIX: observability FIRST, then re-fit via recipe/quant/offload (not higher res).
WIRING: canvas is a `render_driver` OVERRIDE (clamped 512x288, `render_driver.py:1165-1179`), NOT a
node-87 widget. Replace the stale `13688` comment with a link to the generated bakeoff manifest.
ACCEPTANCE: all 3 slots; render-phase NVML <=14.5 GB; audio byte-identical.

## S-C  AUDIO-IN CONDITIONING  [split]
C1 = `audio_motion_profile` extraction + ledger stamp (schema field, producer node, IS_CHANGED/cache
key, PROVE conditioning WAVs never replace the master). C2 = per-engine consumers + HuMo phrase-chunking
(chunk to the cap, mirror-extend) -- also attacks the S-A root. Probe-gated HQ tiers last.

## S-D  gemma normalize_length WRAPPER-KEY DRIFT  [MED]
FIX (CORRECTED location): add `@model_validator(mode="before")` ON `RadioEditPlan` that unwraps EXACTLY
`{"RadioEditPlan": {...}}` and REJECTS ambiguous/multi-key wrappers -- NOT the shared tolerant core
(`_otr_structured_call.py:322-351` -- alias drift is schema-owned there). Add a gemma-shaped regression.

## S-E  NO-FALLBACKS CLEANUP + ENGINE-MENU + UX  [HIGH]
Runtime no-fallback ALREADY enforced (`render_shot` raises LOUD, `render_driver.py:1526-1553`). This
sprint removes scaffolding + UI, as a MIGRATION (not bare deletion):
- `allow_auto_fallback`: a node-87 widget (value `false` in `widgets_values`) + `direct()` signature +
  policy JSON (`otr_video_director.py:216, 278-284, 340-342`). Migrate INPUT_TYPES + signature + policy
  consumers/tests + node-87 `inputs`/`widgets_values` in ONE JSON change + widget audit (POSITIONAL
  drift, BUG-097). Operator's call: DELETE-with-rebaseline vs DEPRECATE-in-place.
- Fallback constants/fns: `make_fallback_of` is STILL CALLED (run_real_episode `render_driver.py:1737`,
  render_single `:2217`); `FLOOR_NAMES`/`UNIVERSAL_FLOOR`/`SYNTH_FALLBACKS` referenced
  (`render_driver.py:46-58,146,157-159,2282`; `eng_character_3d.py:55,257,326`) + asserted in tests
  (`test_video_character_3d.py:363-369`, `test_video_render_driver_additive.py:77-82`). Migrate call
  sites + imports + tests in the SAME commit (or keep no-op dead stubs until tests migrate).
- SEPARATE "remove fallback use" from "remove SELECTABLE engines": `still_motion`/`station_card`/
  `abstract` are registered engines w/ default roles + capability rows
  (`cheap_families.py:165-190`, `registry.py:127-133`). Unregister + update defaults/profiles/tests +
  node-87 dropdowns explicitly.
- DROPDOWN LABELS: current = `engine_id`+aspect (`otr_video_director.py:61-63,105-118`); registry has no
  display/recipe field (`registry.py:127-232`). Add a registry display-metadata field; generate labels
  (model+variant+recipe+VRAM tier; HuMo 1.7B LOW ~3.3 GB / 14B HIGH ~15.9 GB; KEEP BOTH).
- RECIPE-STAMP (CORRECTED): extend `meta.render_engines` (already saved + durable --
  `otr_video_render_batch.py:26,49,227`; `meta` is per-key merged by `_merge_with_disk`), do NOT touch
  `TOP_PRESERVE` (would preserve stale `video.shots` wholesale). PREREQ: engines must RETURN
  `recipe/unet/lora/quant/canvas/audio_source/phase` in the raw/canonical clip (`eng_ltx_av` returns
  only `out_path/frame_count/vram_peak_mb`; `_clip_from_raw` keeps only `vram_peak_mb`,
  `eng_ltx_av.py:621-627,691-694,776-788`) -> thread into `build_clip_manifest` -> into the
  `meta.render_engines` payload.
- ANNOUNCER + MUSIC = always a radio-themed still (vintage radio; never black/abstract/fallback).
  `otr_meta_brief_image_prompt.py` (broadcast-studio vocab already at :127).
- ADD `visualizer_rainbow` -- DEFERRED (own `/roundtable` for the GLSL shader stack first).

## BUG-411  flux CINEMATIC RESTORE  [look-QA verify -- mostly DONE]
CHECKLIST (current truth grounded): FluxGuidance @3.5 DONE (`flux_gen1.py:88-92,130-135`); cinematic
grade tail DONE (`otr_meta_brief_image_prompt.py:535`); radio broadcast-distress tail DONE (:805);
portrait STYLE_ANCHOR DONE (:92). VERIFY only the bookend seed 4242; implement ONLY if missing. No JSON
node (image gen runs through `OTR_ImageGenDispatcher`; Flux builds its own in-process graph).

## OPEN FOR r3 (wiring)
1. allow_auto_fallback: DELETE-with-JSON-rebaseline vs DEPRECATE-in-place (positional widget drift).
2. S-F: exact constant-producer node insertion + which upstream links to sever, validated against the
   real `otr_scifi_16gb_full.json` link list.
3. The full node-87 `widgets_values` positional map BEFORE/AFTER the engine retirements + label field +
   allow_auto_fallback change (one JSON migration, OTR_WorkflowValidator green).
4. recipe-stamp: the exact `meta.render_engines` payload shape + the engine-return plumbing per engine.

(ComfyUI node-class / tensor / VRAM / IS_CHANGED / import-isolation profile invariants still apply.)
