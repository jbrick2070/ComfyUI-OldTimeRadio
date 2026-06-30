# OTR COVERAGE-SOAK FIX SPRINT -- WIRE-READY PLAN (r1-hardened)

Improved after kibitz r1 (Codex/gpt-5.5 panel + Claude code-grounded judging; Antigravity out of
credits -> Codex-only panel). Every change below is grounded against the real Windows files.

## r1 JUDGMENT (what changed and why)
- ACCEPTED (verified): BUG-411 FluxGuidance is ALREADY done; S-E runtime no-fallback is ALREADY done
  (only dead scaffolding remains); S-F seam scripts do not exist; S-A has a split frame-count contract;
  ltx_audio_in canvas is a render-driver override not a node-87 widget; retire-targets double as
  FLOOR_NAMES; labels need a new registry field; recipe-stamp is video-side only (image already durable).
- VERIFY-AT-BUILD: the remaining BUG-411 levers (suffixes/seed) in otr_meta_brief_image_prompt.py;
  exact node 89-92 I/O for the S-F injection; whether any fallback TEST asserts behavior vs the dead
  helpers.

## INVARIANTS (a "fix" that breaks one is rejected)
- `workflows/otr_scifi_16gb_full.json` is the production workflow: any node/wiring/widget change goes IN
  it in the SAME change as the code; re-validate (`OTR_WorkflowValidator` + JSON round-trip + link/widget
  audit); `widgets_values` POSITIONAL (append-only).
- Single resident heavy <= 14.5 GB; 100% local/offline; seed-keyed determinism; UTF-8 no BOM; SFW; no
  "dummy".
- Master audio BYTE-IDENTICAL (`test_audio_byte_identical` GREEN); audio spine FROZEN; conditioning WAVs
  are model-input ONLY.
- NO FALLBACKS (operator 2026-06-29) -- already enforced at render time (see S-E); a selected engine
  RENDERS or HARD-FAILS LOUD.
- Suite + Bug Bible + B7 after every change; commit AND push per green chunk to `v2.0-alpha`; verify
  HEAD==origin / no 0-byte / no BOM / AST parse.

## CORRECTION (do NOT chase): HuMo "mush" is NOT a cfg regression
`eng_humo.py` cfg defaults are deliberate + measured (14B distill 1.0; 1.7B 1.0 de-blued 2026-06-17 was
5.0; 1.7B_169 2.5). Motion is AUDIO-driven, not cfg. Real cause = the clip underrun (S-A).

## RECOMMENDED ORDER
S-F first (accelerator) BUT bake the fixture from a CLEAN short-beat reference episode (no underrun) so
the baseline is not a defect -- OR land S-A first. Then S-A, S-B, S-D, S-E core, S-C (split), BUG-411
(parallel). `visualizer_rainbow` deferred (own roundtable).

---

## S-F  VISUAL SMOKE FIXTURE  [ACCELERATOR]
GOAL: test visual engines without re-running writer + audio per leg.
SEAM (CORRECTED): the named `_otr_soak_phase0/1.py` DO NOT EXIST -- do not cite them. Build on the REAL
harness: `scripts/otr_coverage_sweep.py` + `scripts/_otr_soak_capstone.py` + `scripts/otr_api.py` (the
current runner executes a FULL 30-word episode through writer/audio/video).
INJECTION (CORRECTED -- pick ONE, do not leave open):
  (A) API/workflow PATCH: feed the existing upstream node inputs from a baked fixture -- node 91
      (`script_json`, `image_policy_json`, `image_prompts_json`, `audio_done`, `episode_id`) and node 92
      (`patched_ledger_json`, `master_audio_path`, `image_done`) -- and INVALIDATE the writer/audio node
      cache so they do not re-run; OR
  (B) an explicit fixture-loader NODE wired into the production JSON per repo rules.
BAKE: one good 30-word episode's master audio + story ledger (cast/brief/beats/durations/portrait
hashes), from a CLEAN reference (no underrun).
ACCEPTANCE: a fixture leg renders stills->video->composite->mux with NO writer/audio node execution
(prove via trace), audio byte-identical to the baked master, time = render tail only. Stamp a fixture
HASH on every leg report so eyeball comparisons prove the same baked input.
CONSTRAINT: TEST harness; the production workflow still writes a fresh story per real episode.

## S-A  CLIP-FILL + LEGIBILITY FLOOR  [HIGH]
ROOT CAUSE (grounded): `humo_1.7B` underruns (177-frame ceiling vs 434-frame beat); the 14B is capped at
49 frames (`eng_humo.py:61`). `otr_silent_composite.py:243` `_warn_clip_underrun` ALREADY logs the
underrun LOUD at 50% (`_CLIP_UNDERRUN_FRAC=0.5`) -- detection exists; the FILL does not: short clips are
encoded with `tpad=clone` (a last-frame hold, ~otr_silent_composite.py:395-411).
ONE CONTRACT (CORRECTED -- resolve the split): the engine side mirror-extends capped renders
(`eng_humo.py:102-107`) but the composite still holds. PICK ONE: either every motion engine RETURNS
`frame_count == target` (mirror/loop in the engine), OR `OTR_SilentComposite` LOOPS / ping-pongs any
short real clip (never `tpad=clone`). Recommend: composite-side fill (engine-agnostic, covers every
engine). Replace the `tpad=clone` hold with a boomerang/loop extender.
LEGIBILITY GUARD: after each clip, sharpness RATIO vs source + motion via `freezedetect`; on fail HARD
FAIL LOUD (no fallback), restamp `delivered_engine`/`fail_reason`.
ACCEPTANCE: assert the DELIVERED frame count in the manifest == target (not just "no warning"); continuous
motion to beat end; audio byte-identical.
NOTE: S-C HuMo phrase-chunking attacks the same root from the audio side -- sequence (panel: which is
PRIMARY -- composite loop vs audio chunk vs more frames, given the 49-frame cap + byte-identical audio).

## S-B  ltx_audio_in VRAM FIT  [HIGH]
ROOT CAUSE: regression `7bbce1d8` + `fd9edc28` (dev-Q3_K_M + SHARP, ~15.5-15.9 GB, `eng_ltx_av.py:687`);
last-good `c4d7815b` @ 512x288 = 13688 MB.
FIX: (1) observability FIRST (per-beat recipe/unet/quant/LoRA/canvas/frames/audio-source/phase/peak
VRAM); (2) re-fit via recipe/quant/offload (`OTR_LTX_AV_RECIPE=distilled_native`/lighter quant), NOT
higher res.
WIRING (CORRECTED): the ltx_audio_in canvas is a `render_driver` OVERRIDE (clamped 512x288,
`render_driver.py:1165-1179`), NOT a node-87 widget (node 87 only has GLOBAL `canvas_w/h` 832x480). Fix
in the override (or add a real per-engine canvas control); replace the stale `13688` comment with "see
runtime logs / bakeoff manifest".
ACCEPTANCE: renders in all 3 slots; render-phase NVML <= 14.5 GB; audio byte-identical.

## S-C  AUDIO-IN CONDITIONING  [split -- not a MED bolt-on]
SPLIT (CORRECTED): C1 = `audio_motion_profile` EXTRACTION + ledger stamp (define the ledger schema field,
the producer node/location, the cache key / `IS_CHANGED` behaviour, and PROVE conditioning WAVs never
replace the master); C2 = per-engine CONSUMERS (audio-in engines get real audio; others get
prompt/camera/parallax/light). HuMo phrase-chunking (chunk to the 49-frame cap, mirror-extend per chunk)
lands in C2 and also attacks the S-A root. Probe-gated HQ tiers last.

## S-D  gemma normalize_length WRAPPER-KEY DRIFT  [MED]
ROOT CAUSE: gemma nests RadioEditPlan under a top-level `RadioEditPlan` key -> `projected_word_total`
"missing" -> retry ladder exhausts -> normalization skipped. Fix the LEVER-1 tolerant-unwrap to peel a
top-level schema-name wrapper; retest a gemma leg.

## S-E  NO-FALLBACKS CLEANUP + ENGINE-MENU + UX  [HIGH]
REFRAMED (CORRECTED): the RUNTIME no-fallback is ALREADY DONE -- `render_shot` raises LOUD
"fallbacks are disabled" (`render_driver.py:1526-1553`). This sprint REMOVES the now-DEAD scaffolding (no
render-path behaviour change):
- DELETE the `allow_auto_fallback` widget (`otr_video_director.py:216,282,342`) + schema field
  (`schemas.py:128`, default True) -- it advertises a capability that no longer exists.
- DELETE `FLOOR_NAMES` / `UNIVERSAL_FLOOR` / `SYNTH_FALLBACKS` + their references
  (`render_driver.py:46-58,146,157-159,2282`; `eng_character_3d.py:55,257,326`) and the now-stale
  fallback tests (`test_video_fallback_chain_additive.py`, `test_video_retry_taxonomy.py`, ...).
- DISAMBIGUATE retry vs fallback: KILL cross-engine fallback (done + scaffolding removed); KEEP a bounded
  same-engine RETRY (`retry_taxonomy.py`) -- do not gut the retry taxonomy.
- SEQUENCE: remove refs -> delete constants -> delete widget/schema -> unregister
  `still_motion`/`station_card`/`abstract` (which DOUBLE as `FLOOR_NAMES`, `cheap_families.py:165-191`;
  capability rows `registry.py:127-139`) -> remove node-87 JSON dropdown values -> update tests.
- DROPDOWN LABELS (CORRECTED): current labels = `engine_id` + aspect suffix
  (`otr_video_director.py:61-63,105-118`); the registry has NO display/recipe field
  (`registry.py:127-232`). ADD a registry display-metadata field and GENERATE labels from it. State
  model+variant+recipe+VRAM tier (HuMo: 1.7B LOW-VRAM ~3.3 GB fast draft / 14B HIGH-VRAM ~15.9 GB max
  quality, spills 16 GB -- KEEP BOTH).
- RECIPE-STAMP (CORRECTED -- video side only): the IMAGE engine is ALREADY durable in `ledger['images']`
  (`otr_video_render_batch.py:31`). Add the per-beat VIDEO `delivered_engine` + `recipe/quant`, and make
  it DURABLE through `_merge_with_disk` -- add the key to `TOP_PRESERVE`
  (`production_ledger.py:1220`, today only `schema_version/audio_gates/transitions/radio_bookend_path`).
- ANNOUNCER + MUSIC = always a radio-themed still (vintage radio in scene); never black card/abstract/
  fallback. Image-prompt change (`otr_meta_brief_image_prompt.py`).
- ADD `visualizer_rainbow` (DEFERRED -- after correctness; own `/roundtable` for the GLSL shader stack
  first). Reuses `eng_visualizer.py` audio analysis. Register + CAPABILITIES + node-87 dropdown + label.

## BUG-411  flux CINEMATIC RESTORE  [parallel -- mostly DONE]
CUT (CORRECTED): FluxGuidance @ 3.5 is ALREADY implemented (`flux_gen1.py:88-92` `OTR_FLUX_GUIDANCE`
default 3.5; wired as a `FluxGuidance` node `flux_gen1.py:130-135`). No JSON node needed (image gen runs
through `OTR_ImageGenDispatcher`; Flux builds its own in-process graph).
REMAINING (VERIFY each in `otr_meta_brief_image_prompt.py` -- may also be done): cinematic style suffix,
radio broadcast-distress suffix, bookend seed 4242, portrait style line. Only restore the ones genuinely
missing. Co-schedule with the S-E radio-still bookend (same prompt file -- one window).

## OPEN QUESTIONS FOR r2 (coding)
1. S-A primary: composite loop/ping-pong vs audio phrase-chunk vs render-more-frames?
2. S-F: injection (A) API-patch vs (B) fixture-loader node -- which is cleaner given cache invalidation +
   the `_merge_with_disk` image-drop?
3. S-E: full reference list that still names `FLOOR_NAMES`/`SYNTH_FALLBACKS` so deletion leaves the suite
   green.

(ComfyUI node-class / tensor / VRAM / IS_CHANGED / import-isolation invariants from the profile still
apply -- check new/edited nodes against them.)
