# OTR COVERAGE-SOAK FIX SPRINT -- WIRE-READY PLAN (kibitz-CONVERGED 2026-06-29)

Hardened across a full 4-round kibitz arc (Codex/gpt-5.5 panel + Claude code-grounded judging;
Antigravity out of credits -> Codex-only panel). r4 verdict: CONVERGED. Every claim below is grounded
against the real Windows files. This is the coder-window contract; raw rounds + judgments in
`kibitz-runs/2026-06-29-otr-soak-fixes/`.

## INVARIANTS (a fix that breaks one is rejected)
- `workflows/otr_scifi_16gb_full.json` is the production workflow: any node/wiring/widget change goes IN
  it in the SAME commit; positional `widgets_values` (append-only); re-validate (`OTR_WorkflowValidator`
  + JSON round-trip + link/widget audit).
- Single resident heavy <= 14.5 GB; 100% local/offline; seed-keyed determinism; UTF-8 no BOM; SFW; no
  "dummy".
- Master audio BYTE-IDENTICAL (`test_audio_byte_identical` GREEN); audio spine FROZEN; conditioning WAVs
  model-input ONLY.
- NO FALLBACKS -- ALREADY enforced at render time (`render_shot` raises LOUD, `render_driver.py:1526-1553`).
- Suite + Bug Bible + B7 after every change; commit AND push per green chunk to `v2.0-alpha`; verify
  HEAD==origin / no 0-byte / no BOM / AST parse.

## CORRECTION (do NOT chase): HuMo "mush" is NOT a cfg regression
`eng_humo.py` cfg defaults are deliberate + measured (14B distill 1.0; 1.7B 1.0 de-blued 2026-06-17 from
5.0; 1.7B_169 2.5). Motion is AUDIO-driven, not cfg. Real cause = the clip underrun (S-A).

## ORDER
S-F (bake from a CLEAN no-underrun reference) -> S-A -> S-B -> S-D -> S-E (no-fallback cleanup, THEN
engine-retirement, THEN labels/recipe-stamp/radio-still) -> S-C (split, deferred tail) -> BUG-411 (look-QA
verify). `visualizer_rainbow` deferred (own `/roundtable`).

---

## S-F  VISUAL SMOKE FIXTURE  [ACCELERATOR]
GOAL: test visual engines without re-running writer + audio per leg.
MECHANISM: submit a PRUNED ComfyUI API prompt of ONLY the render-tail nodes -- ComfyUI executes EVERY
OUTPUT_NODE in a prompt, so the full graph would re-pull writer+audio (MasterAudioMux/SceneSequencer/
SceneAwareScopes/PostUpscaleProcgenBlend are all OUTPUT_NODEs). DIRECTLY PATCH node-92
(`OTR_VideoRenderBatch`) converted API-prompt inputs to baked literals -- `patched_ledger_json`,
`master_audio_path`, `image_done` (forceInput defaults `otr_video_render_batch.py:77-99`; empty/invalid
fixture JSON fails :173-181). Do NOT add constant producer nodes (`otr_api.py` converts widgets/links
only). Build on `otr_coverage_sweep.py` + `otr_api.py` + the ComfyUI MCP (submit prompt + read /history).
BAKE A BUNDLE (not just audio+ledger): the ledger + master audio + EVERY ledger-referenced portrait /
scene still / mesh-fodder asset (`build_request_from_shot` resolves them `render_driver.py:842-944`;
`ltx_audio_in` fails loud without the init image `eng_ltx_av.py:629-635`). REWRITE ledger asset paths to
the bundle, and PREFLIGHT `Test-Path` + hash each referenced asset before submitting. Use a CLEAN
reference episode (no underrun).
ACCEPTANCE: the `/history` executed-node set is EXACTLY `{92}` (or `{63, 92}` if the validator runs --
node 63 `OTR_WorkflowValidator` is itself OUTPUT_NODE, `_otr_workflow_validator.py:197-198`); writer
node 1 + audio nodes ABSENT; all 6 beats render; baked master audio hash unchanged before/after; time =
render tail; stamp a fixture HASH per leg.

## S-A  CLIP-FILL + LEGIBILITY FLOOR  [HIGH]
ROOT CAUSE: clips underrun the beat; the composite holds the last frame (`tpad=clone`).
`_warn_clip_underrun` (`otr_silent_composite.py:243`) already detects. CAP: `_HUMO_MAX_FRAMES=177` for
base humo/1.7B; only `humo_14B_169` caps at 49 -> fill is SIZE-AGNOSTIC (`frame_count < target`).
FIX: in `plan_timeline_segments` compute `should_loop` (real clip row: `exists && path && frame_count>0
&& frame_count < target`) BEFORE `_warn_clip_underrun` so the LOUD warning stops once fill is on
(`otr_silent_composite.py:243-257, 325-339`); set `loop=True` for those rows (reuse the credits-tail loop
path); keep `tpad`/`loop=False` for floor/credits/non-loop sources (`:360-368, 395-411, 645-649`).
MANIFEST CONTRACT (r4): KEEP raw `frame_count` = engine output (`build_clip_manifest`,
`render_driver.py:2000-2008`) so the loop decision still fires. Add a SEPARATE `delivered_frame_count` /
`segment_frame_count` check AFTER `plan_timeline_segments`/`assemble_silent_timeline`; put `freeze_score`
/ `quality_status` / `fail_reason` in the COMPOSITE report / a post-assemble QA artifact -- do NOT
overwrite raw manifest semantics.
LEGIBILITY GUARD (STAGED): FIRST commit = assert DELIVERED segment frames == target + `freezedetect` on
the SILENT video only (never the master). DEFER sharpness-ratio gating to a later commit.
ACCEPTANCE: delivered segment/output frames == target after loop-fill; raw `frame_count` stays engine-
produced; `test_audio_byte_identical` + composite tests green.

## S-B  ltx_audio_in VRAM FIT  [HIGH]
ROOT CAUSE: regression `7bbce1d8`+`fd9edc28` (~15.9 GB, `eng_ltx_av.py:687`); last-good `c4d7815b`
@512x288=13688 MB. FIX: observability FIRST (per-beat recipe/unet/quant/LoRA/canvas/frames/audio-source/
phase/peak VRAM), then re-fit via recipe/quant/offload (not higher res).
WIRING: canvas is a `render_driver` override, env-overridable via `OTR_LTX_AV_RENDER_CANVAS`
(applied `render_driver.py:1173-1179`), NOT a node-87 widget. ACCEPTANCE: explicitly SET + RECORD (in the
bakeoff/run manifest) the effective `OTR_LTX_AV_RENDER_CANVAS=512x288` + the canvas applied in
`build_request_from_shot` + measured NVML peak <= 14.5 GB across all 3 slots; audio byte-identical.
Replace the stale `13688` comment with a bakeoff-manifest link.

## S-D  gemma normalize_length WRAPPER-KEY DRIFT  [MED]
FIX: add `@model_validator(mode="before")` on `RadioEditPlan` (`_otr_radio_editor.py:316-323`, NO
before-validator today) that unwraps EXACTLY `{"RadioEditPlan": {...}}` and REJECTS ambiguous/multi-key
wrappers -- NOT the shared tolerant core. Add the multi-key-wrapper regression in the SAME commit.

## S-E  NO-FALLBACKS CLEANUP + ENGINE-MENU + UX  [HIGH] -- ordered sub-steps
E1. NO-FALLBACK SCAFFOLDING (runtime already loud): `make_fallback_of` is STILL CALLED (run_real_episode
   `render_driver.py:1737`, render_single `:2217`); migrate those call sites + the constants
   (`FLOOR_NAMES`/`UNIVERSAL_FLOOR`/`SYNTH_FALLBACKS`/`ENGINE_FAMILY`/`_PROFILES`/`EXPECTED_OOM_TRAIL`,
   `render_driver.py:46-107,146,157-159,2282`; `eng_character_3d.py:55,257,326`) + the fallback tests
   (`test_video_character_3d.py:363-369`, `test_video_render_driver_additive.py:77-82`) in ONE commit.
E2. `allow_auto_fallback` = DEPRECATE-IN-PLACE (NOT delete this sprint): keep node-87 input/widget (wv13)
   + `direct()` signature (`otr_video_director.py:216,278-283,342`), force it false in the emitted policy,
   relabel "(deprecated)". Deletion shifts wv14..18 and `otr_api.py:608-617` hard-fails on widget-length
   drift -- do a clean delete only as a SEPARATE JSON rebaseline.
E3. ENGINE RETIREMENT (SEPARABLE / can defer -- bigger than dropdowns): `still_motion`/`station_card`/
   `abstract` live in the floor constants above + `cheap_families.py:165-190` + capability rows
   `registry.py:127-133` + soak-fixture expectations + tests. Do this AFTER E1 (the floor constants are
   gone by then). Migrate every remaining runtime constant / soak fixture / capability row / test, then
   unregister + remove node-87 dropdown options (current saved node-87 values are
   visualizer/flux_gen1/humo_14B_169 -- none is a retired engine, so NO widget-value rewrite needed).
E4. DROPDOWN LABELS: MUST stay `engine_id (display metadata...)` -- `_engine_id_from_pick` parses text
   BEFORE the first `" ("` (`otr_video_director.py:61-87,105-125`). Add a registry display field; render
   e.g. `humo_1.7B (HuMo 1.7B, portrait, LOW-VRAM ~3.3 GB)`, `humo (HuMo 14B, portrait, HIGH-VRAM
   ~15.9 GB)` -- KEEP BOTH HuMos. Add a test: every `_video_model_combo()` label round-trips through
   `_engine_id_from_pick`.
E5. RECIPE-STAMP: extend the EXISTING versioned `meta.render_engines` payload (today: histogram /
   video_revision / by_role / vram_peak_mb; `otr_video_render_batch.py:26-49`) -- PRESERVE those keys, add
   a `per_clip`/`by_engine` recipe block (`delivered_engine` + recipe/quant); engines without recipe
   fields -> `recipe=null` (never drop the existing keys). PREREQ: engines must RETURN
   recipe/unet/lora/quant/canvas/audio_source/phase in the raw clip (`eng_ltx_av` returns only
   out_path/frame_count/vram_peak_mb; `_clip_from_raw` keeps only vram_peak_mb,
   `eng_ltx_av.py:621-627,691-694,776-788`) -> thread into `build_clip_manifest` -> the payload. `meta`
   is per-key merged by `_merge_with_disk` (durable); do NOT touch `TOP_PRESERVE`.
E6. ANNOUNCER + MUSIC = always a radio-themed still (broadcast-studio vocab already at
   `otr_meta_brief_image_prompt.py:127`); never black/abstract/fallback.
E7. ADD `visualizer_rainbow` -- DEFERRED (own `/roundtable` for the GLSL shader stack first; reuses
   `eng_visualizer.py` audio analysis).

## S-C  AUDIO-IN CONDITIONING  [split -- deferred tail]
C1 = `audio_motion_profile` extraction + ledger stamp (schema field, producer node, IS_CHANGED/cache key,
PROVE conditioning WAVs never replace the master). C2 (deferred) = per-engine consumers + HuMo
phrase-chunking. Not needed to land S-A/S-B/S-F.

## BUG-411  [look-QA verify -- DONE except one check]
FluxGuidance @3.5 (`flux_gen1.py:88-92,130-135`), cinematic grade tail (`otr_meta_brief_image_prompt.py
:535`), radio distress tail (:805), portrait STYLE_ANCHOR (:92) ALL present. VERIFY only the bookend
seed 4242; implement only if missing. No JSON node (image gen runs through `OTR_ImageGenDispatcher`).

## VERIFY-AT-BUILD CHECKLIST (r4)
1. S-F: pruned prompt renders all beats from the baked BUNDLE, no missing-dependency error.
2. S-F: `/history` executed set == `{92}` or `{63,92}`; writer node 1 + audio nodes absent.
3. S-F: baked master audio hash unchanged before/after; every bundle asset preflight-exists + hash-matches.
4. S-E: `allow_auto_fallback` stays in node-87 widgets/signature, policy forces false, `otr_api.py`
   conversion green.
5. S-E: every displayed label parses back to a registered engine id via `_engine_id_from_pick`.
6. S-A: raw `frame_count` stays engine-produced; delivered segment frames == target after loop-fill;
   `test_audio_byte_identical` + composite tests green.
7. S-B: `OTR_LTX_AV_RENDER_CANVAS=512x288` set + recorded; NVML peak <= 14.5 GB across all 3 slots.

(ComfyUI node-class / tensor / VRAM / IS_CHANGED / import-isolation profile invariants apply to every new
/ edited node.)
