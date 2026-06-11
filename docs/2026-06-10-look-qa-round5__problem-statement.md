# Look-QA Round 5 -- the acceptance eyeball FAILED; forensic root causes + fix plan (DRAFT for roundtable)

Date: 2026-06-10 evening. Episode: *Symphony of Silicon* (`signal_lost_symphony_of_silicon_20260610_155504`),
the post-gap-audit 30w acceptance render from the SAVED `workflows/otr_scifi_16gb_full.json`
(words=30 the only patch), HEAD `56caa5b` (7 unpushed), suite 3815/0 + Bug Bible green,
ALL nine LOG gates green (3x dispositions, 3x brief-composed LTX SCENE lines, distinct
portrait hashes, captions 7 events, credits 843->1029, duration/byte-identical/obs OK,
histogram ltx:3/humo:3, zero fallbacks, VRAM peak 4.3GB).

**The operator eyeball still FAILED**: (1) no LTX radio visual at the start, (2) the LTX
scenes look bad, (3) beats without people ("non-people portraits"), "I don't see the change."
The log gates measured that the new code RAN; they did not measure that it produced the
LEGACY LOOK. Forensics below; every claim has a evidence pointer.

## Operator constraints (verbatim rules that bind every fix below)

- Audio is SHIPPED & FROZEN: ledger read-only, `test_audio_byte_identical` green, mux-LAST.
- No other sprints (3D / switchable-workflow / whiny-voice / story-* all PARKED).
- 7 commits unpushed on v2.0-alpha; operator pushes after a PASSING eyeball. Do not push.
- Single resident heavy engine <= 14.5GB NVML; BUG-291 reclaim; LOUD fallbacks; UTF-8 no BOM; SFW.
- Fixes must land in the SAVED-workflow production path (no headless-only patches -- that
  drift class is what the punch list killed).

## Root causes (evidence-grade)

### D1 -- The opening is featureless mud (operator: "no ltx radio visual to start")
- The synthetic `b000_music_open` covers the FULL head gap: 9.5s -> `target_frame_count`
  ~238 @ 25fps (captions place first speech at 9.50s).
- `eng_ltx_video._render_in_process` passes `plan["target_frame_count"]` STRAIGHT to the
  sampler -- **no frame cap** (`eng_ltx_video.py` ~line 280). The proven-good LTX renders
  are 49-121f; 238f @ 1472x832 is far past LTX coherence and the clip collapses into a
  flat brown field.
- Proof: per-second luma of the SILENT composite -- YAVG is a CONSTANT ~35.8-36.3 for
  t=0..9 (a textureless field; video black would be ~16), vs 21.7 (announcer scene),
  30-37 (cast beats). The composite DID place the clip ("assembled 7 beats"); the source
  render itself is mud. The b000 sampler also ran ~12s/it vs 7.2/5.3 for b001/b005 --
  the frame count was the outlier.
- NOT the cause: composite placement (7 beats OK), the prompt (identical to b001's which
  rendered a real scene), the blend (mud in upstream).

### D2 -- The LTX scenes "look bad": one terse dark prompt, reused three times
- All three LTX beats logged the IDENTICAL prompt: `(236 chars) "Innovator presents
  groundbreaking machine, colleague expresses unease, a vintage radio set glowing..."` --
  `get_story_brief_ltx(meta)` is ONE logline per EPISODE, so every text-engine shot in the
  episode shares it. Era tail = this episode's brief atmosphere terms (dusk-dark for this
  brief). Result: announcer scene = small lit figure in a dark room (silent-composite
  luma 21.7, the darkest stretch of the episode), buried further under the green HUD blend.
- The LEGACY pipeline (preserved at `docs/2026-06-10-brief-downstream-gaps/legacy_otr_video_plan_e74a3ce.py.txt`,
  the file the operator asked to compare): per-shot prompt =
  `truncated_portrait_prompt + scene.visual_prompt + shot_hint + era_tail + style_tail`
  (compose_shot_prompt, ~line 380-403). `scene.visual_prompt` came from the WRITER's
  per-scene derivation (PASS 2) -- varied, episode-specific, usually peopled and lit
  ("warm tungsten light" templates). The gap-audit restoration (c51526b) revived the
  TAILS and the brief logline core but NOT the per-scene visual_prompt layer -- the
  actual source of the old scenic varied opens.
- KEY FACT: tonight's ledger has `meta.visual_plan` PRESENT (the writer still emits it;
  pass00 amendment said radio editor / casting / HUD genre consume it) -- the new LTX
  branch just never reads it.
- This is why the operator "doesn't see the change": the restored code RAN (dispositions
  prove it) but restored a thinner composition than the legacy look.

### D3 -- Beats without people / wrong faces
- The three FLUX portraits ARE all people and in-character (foundry innovator with mic,
  standing engineer, crucible man) -- the dae597a portrait person-guard did its job.
  `output\otr\stills\{6641f7d4,a319a235,f0311c51}*.png` = c01/c02/c03.
- **b002 (HAYES, "It's alive, Gulliver")**: the CORRECT portrait was staged into ComfyUI
  input at 15:56:37 (`a319a235...png` = c02). The rendered HuMo clip is the foundry
  CONSOLE with NO person (silent t=16.8). The line's RAW text was "(fingers dancing on
  the console) It's alive..." -- the stage-direction scrub cleaned the SPOKEN text, but
  the M4 batch-LLM still described the console ACTION in the shot's `text_prompt`, and
  HuMo (text+image+audio) followed the text away from the init face. There is NO
  person-anchor requirement on M4 SHOT prompts (the person guard exists only for FLUX
  portrait prompts).
- **b003+b004 share one face for two speakers**: the writer attributed b004
  ("Gulliver, it's not just a machine...") to char_id=c03 (GULLIVER) -- but the text is
  HAYES addressing Gulliver. A self-vocative ("Gulliver," spoken by Gulliver) that the
  a5f4763 pre-freeze self-vocative scrub did NOT catch (no scrub log for b004; only the
  b002 stage-direction scrub fired). Wrong attribution -> both beats animate c03's
  portrait. The render layer behaved correctly.
- Noise also found: cast row `HAYES VANCE gender=female` with a male portrait (the known
  F0-auto mislabel class; whiny-voice territory, PARKED -- list, don't fix here).
- Latent join hazard: announcer LINES carry `char_id='announcer'` while cast/portraits
  key him `c01` -- harmless today (announcer beats are LTX) but a face-engine remap away
  from a silent portrait miss. Flag for a cheap LOUD warning.

### Ruled OUT tonight
- Workflow-JSON drift (the operator's first suspicion): the visual-structure pin test is
  green, captions/credits/LTX-open/Director-roles all fired from the SAVED json, and the
  engine histogram came out exactly as ordered (ltx:3/humo:3, zero fallbacks). The saved
  workflow IS current. The misses are node-code seams + writer attribution.
- Engine routing, portrait distinctness, caption/credits plumbing, audio integrity: all
  verified good (gates + ffprobe aac/h264/1472x832).

## Proposed fixes (R5-1..R5-6) -- panel: harden, re-scope, or strike

- **R5-1 (D1, render)**: cap text-engine frame asks at the adapter seam:
  `OTR_LTX_MAX_FRAMES` default 121 (or 161); when a shot's window exceeds the cap, render
  the cap and FILL the remainder in the composite by hold-last-frame with the existing
  slow-zoom (Ken Burns) treatment, never stretch-interpolate. LOUD log when capping.
  (Alternative considered: split long synthetic beats at ShotLock into N chained shots --
  more render cost, more seams; panel may prefer it for motion variety.)
- **R5-2 (D2, prompts)**: restore the legacy per-scene layer: LTX scene core becomes
  `scene_visual_prompt (from meta.visual_plan, per-beat/scene) else get_story_brief_ltx`,
  still finished with the era tail under the 240 cap. Per-beat variety returns; the brief
  logline stays as the fallback. Map beats->scenes via the visual_plan's scene index
  (fall back to scene 0 / nearest).
- **R5-3 (D2, opens)**: bias the OPEN clauses bright: replace the bare "a vintage radio
  set glowing in the scene" with "a vintage radio set glowing warmly, lit dials and
  tubes" + keep drift/no-text clauses. (Operator look call; cheap string change, panel
  judges wording only.)
- **R5-4 (D3, M4 person anchor)**: the M4 batch-LLM instruction for TALKING-HEAD shots
  gains a hard requirement: the prompt MUST describe the named character as the visible
  subject (face-forward, mid-shot or closer); a post-guard rejects/repairs M4 prompts for
  humo-routed shots that lack a person token (mirror of the FLUX person guard, prompt-text
  level, CPU-cheap, fail-soft to the portrait-anchored template).
- **R5-5 (D3, attribution)**: extend the self-vocative scrub to the ATTRIBUTION level:
  when line N's text begins with the addressee-vocative of the SPEAKER's own name
  ("<OwnName>, ..."), flag and re-attribute via the LLM repair pass (or at minimum LOUD-log;
  silent wrong faces are how tonight slipped). Writer-side, CPU tests.
- **R5-6 (D3, join hazard)**: LOUD warning when a TALKING-HEAD shot's char_id misses the
  portrait index (today it silently fails-closed only at eng_humo); plus normalize the
  announcer line char_id to the cast row id (c01) at ShotLock so the join is uniform.

## Invariants for the panel
Frozen audio untouched (all fixes are video/prompt/writer-side; the writer fix R5-5 runs
BEFORE the freeze in future episodes, never rewrites a frozen ledger); fail-soft never
fail-episode; explicit env overrides verbatim; no new widgets; the SAVED workflow is the
only path; UTF-8 no BOM; SFW; suite + Bug Bible + byte-identical green at every commit.

## Acceptance (the re-render gate)
ONE fresh 30w from the saved workflow (words=30 only): b000 window shows a REAL scene
(luma variance, not a flat field; no >cap LTX asks in the log), per-beat LTX prompts
DIFFER (b000/b001/b005 not identical), every humo beat shows its OWN cast face
(person visible at beat midpoints), no self-vocative/mis-attributed line ships, all nine
log gates stay green, audio byte-identical, operator eyeball PASSES.
