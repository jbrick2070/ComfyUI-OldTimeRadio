# OTR COVERAGE-SOAK FIX SPRINT -- WIRE-READY PLAN (2026-06-29)

Single consolidated plan for every fix the 30-word coverage soak surfaced (every video + image
engine x 3 dropdown slots, writer = local gemma). Target for a kibitz hardening pass, then a fresh
coder window executes it. Keep it LEAN and wire-ready.

- Branch: `v2.0-alpha`. Workflow of record: `workflows/otr_scifi_16gb_full.json` (node-87
  `OTR_VideoDirector` carries the slot dropdowns).
- Two dropdown families per beat-role: an IMAGE engine (mints the still) and a VIDEO engine
  (animates it). Three video slots: `announcer_visual` / `music_visual` / `other_beats_visual`.

## INVARIANTS (apply to EVERY sprint -- a "fix" that breaks one is rejected)

- `otr_scifi_16gb_full.json` is the production workflow: ANY node / wiring / widget change goes IN
  it in the SAME change as the code (unwired code is dead). Re-validate after: `OTR_WorkflowValidator`
  + JSON round-trip + link/widget audit. `widgets_values` is POSITIONAL -- only APPEND optional
  widgets at the end.
- Single resident heavy engine <= 14.5 GB (host NVML); 100% local / offline; determinism seed-keyed;
  UTF-8 no BOM; SFW; never "dummy" (use placeholder/stub).
- Master audio BYTE-IDENTICAL (`test_audio_byte_identical` stays GREEN); the audio spine is FROZEN
  (mux-LAST, no `-shortest`). Any conditioning WAV is model-input ONLY -- never touches the master.
- NO FALLBACKS (operator 2026-06-29): a selected engine RENDERS or HARD-FAILS LOUD with a named
  reason. No silent degrade to stills.
- Run the regression suite + Bug Bible + the B7 forbidden-sweep after EVERY code change; commit AND
  push per green chunk to `v2.0-alpha`; verify HEAD==origin / no 0-byte / no BOM / AST parse.

## CORRECTION baked in -- do NOT chase a HuMo cfg regression

HuMo "mush" is NOT a cfg regression. `eng_humo.py` cfg defaults are deliberate + measured: 14B
distill = cfg 1.0 (`_cfg`); 1.7B portrait = cfg 1.0 (de-blued 2026-06-17, was 5.0 -- 5.0 measured a
strong blue cast, B-R +44.6 vs the source still's +25); 1.7B_169 = cfg 2.5 (sweep sweet spot); 14B_169
inherits 1.0. The header comment states HuMo motion is AUDIO-driven, not cfg. The real cause of the
soft/frozen picture is the CLIP UNDERRUN (S-A). Any panelist who proposes "raise HuMo cfg" is wrong --
verify against `eng_humo.py` `_cfg()` methods.

## RECOMMENDED ORDER (panel: confirm or re-sequence)

S-F FIRST (accelerator -- collapses every other sprint's test loop), then S-A, S-B, S-D, S-E core,
S-C, then BUG-411 (parallel look-QA track). `visualizer_rainbow` is a creative sprint scheduled AFTER
the correctness fixes (own roundtable first).

---

## S-F  VISUAL SMOKE FIXTURE  [ACCELERATOR -- do FIRST]

GOAL: test visual engines without re-running the writer + audio on every leg.

WHY: today every coverage leg re-runs gemma (minutes) + the full audio path (TTS + music) just to
exercise a VISUAL engine -- most of the ~28 min/leg, and a DIFFERENT story each run (not
apples-to-apples).

DESIGN: BAKE one good 30-word episode's master audio + story ledger (cast / brief / beat structure /
per-beat durations / portrait hashes) ONCE into a fixture; INJECT it and run only
stills -> video -> composite -> mux. Each engine test then swaps ONLY the image/video engine.

SEAM: the MIRROR of the existing audio-only soak, which PRUNES the graph at node-7 `EpisodeAssembler`
to skip video (`scripts/_otr_soak_phase0.py` / `_otr_soak_phase1.py`). This starts FROM that boundary
(frozen audio + `/otr/latest_ledger`) and skips the writer (node 1) + audio nodes.

PAYOFF: ~28 min/leg -> just the render tail; identical story+audio every run = clean per-engine
eyeball; master audio byte-identical for FREE (same baked WAV).

CONSTRAINT: TEST harness ONLY -- no production-path change; the real workflow still writes a fresh
story each episode. Inject via profile/role overrides + a fixture loader, not by editing the
production graph.

ACCEPTANCE: a fixture leg renders stills+video+composite+mux with NO writer/audio node execution
(prove via the run trace), audio byte-identical to the baked master, total time = render tail only.

OPEN: where exactly to inject the frozen ledger so the still/video stages consume it unchanged
(portrait hashes + per-beat durations must survive `production_ledger._merge_with_disk`, which today
drops the top-level `images`). Panel: name the cleanest inject point.

---

## S-A  CLIP-FILL + LEGIBILITY FLOOR  [HIGH]

GOAL: a motion clip that renders fewer frames than its beat must FILL the beat with motion, never a
held / frozen last frame.

ROOT CAUSE (grounded, reproduced on 2 episodes): `humo_1.7B` underruns --
`CLIP UNDERRUN: shot_b005 rendered 177 frame(s) for a 434-frame target (41%); the composite will HOLD
the last frame for the rest of the beat`. 177 = the HuMo per-clip ceiling; the 14B is capped at 49
frames for VRAM safety (`eng_humo.py:61`). The held static last frame IS the dead/mushy plate.
Completion gates (obs ships, audio byte-identical) PASS regardless -- invisible to the harness.

FIX (priority):
1. CLIP-FILL -- a motion engine that underruns LOOPS / ping-pong (boomerang) extends to the target
   frame count (the composite's OWN recommendation), never holds the last frame. Touch the HOLD path
   in `otr_silent_composite.py`.
2. LEGIBILITY GUARD after each clip -- sharpness RATIO vs the source still (relative / catastrophic
   only) + motion via `freezedetect`. On failure -> HARD FAIL LOUD (NO fallback, per S-E), restamp
   `delivered_engine` / `fail_reason` in the ledger. (face-presence check = phase 2.)
3. FORENSIC (aids diagnosis, not the cause): preserve `ledger['images']` durably (today
   `production_ledger._merge_with_disk` drops top-level `images`); stamp per-beat
   `init_image_used` / `init_source`.

ACCEPTANCE: a long-beat HuMo episode shows continuous motion to beat end (`freezedetect` held% under
threshold); no silent still-swap; audio byte-identical.

NOTE: S-C HuMo phrase-chunking attacks the SAME underrun root from the audio side -- sequence so they
do not collide (panel to advise: boomerang-extend vs chunk-the-audio as the PRIMARY).

---

## S-B  ltx_audio_in VRAM FIT  [HIGH]

GOAL: `ltx_audio_in` renders within the 14.5 GB ceiling in all 3 slots (today it hard-fails ~15.9 GB,
`eng_ltx_av.py:687`).

ROOT CAUSE: regression `7bbce1d8` (bakeoff "quality upgrade", PROVISIONAL) + `fd9edc28` (switched to
dev-Q3_K_M + SHARP LoRA, ~15.5 GB). Last-good = `c4d7815b` base recipe @ 512x288 = 13688 MB.

FIX (in order): (1) OBSERVABILITY FIRST -- per-beat log recipe / unet / quant / LoRA / canvas /
frames / audio-source / phase-marker / peak VRAM; (2) RE-FIT via recipe / quant / offload
(`OTR_LTX_AV_RECIPE=distilled_native` / lighter quant), NOT higher resolution; quality/resolution
tiers LAST, probe-gated. Replace the stale `13688` comment in `render_driver.py` with "see runtime
logs / bakeoff manifest".

WIRING: confirm the node-87 ltx canvas widget in the JSON matches the chosen canvas (default
512x288).

ACCEPTANCE: `ltx_audio_in` renders in all 3 slots; render-phase NVML <= 14.5 GB; audio byte-identical.

---

## S-C  AUDIO-IN CONDITIONING  [MED]

GOAL: a shared per-beat `audio_motion_profile` (rms / peak / onset / silence / brightness /
dynamic-range / speech-vs-music / duration) drives EVERY engine -- audio-in engines get real audio;
non-audio engines get prompt / camera / parallax / light from the profile.

NOTES: normalized conditioning WAVs are model-input ONLY (master untouched -> byte-identical holds).
HuMo phrase-chunking for long dialogue (chunk to the 49-frame cap, mirror-extend per chunk) also
attacks the S-A underrun root. Probe-gated HQ tiers last.

---

## S-D  gemma normalize_length WRAPPER-KEY DRIFT  [MED]

GOAL: every gemma episode applies length normalization (today skipped warn-only).

ROOT CAUSE: gemma returns the RadioEditPlan nested under a top-level `RadioEditPlan` key ->
`projected_word_total` reads "missing" -> the retry ladder exhausts -> normalization skipped. Fix the
LEVER-1 tolerant-unwrap to peel a top-level schema-name wrapper; retest on a gemma leg.

---

## S-E  NO-FALLBACKS + ENGINE-MENU + UX CLEANUP  [HIGH] (operator directives 2026-06-29)

- NO FALLBACKS / hard-fail: rip out the fallback chains (`resolve_fallback_chain` / `SYNTH_FALLBACKS`
  / the `humo -> still` degrade). A selected engine RENDERS or raises a LOUD hard error. S-A's
  legibility floor becomes detect-and-FAIL, not a still-swap.
- RETIRE engines: `still_motion` (fallback-floor twin of `still_pan`), `station_card` (broken black
  card, missing `accepts_still`), `abstract` (redundant with `visualizer`). Unregister + remove from
  the node-87 JSON dropdowns + ripple the tests (the C3 rename pattern). `cheap_families.py`.
- DROPDOWN LABELS: every option states model + variant + recipe + VRAM tier. HuMo: 1.7B = LOW-VRAM
  ~3.3 GB fast draft / 14B = HIGH-VRAM ~15.9 GB max quality (spills 16 GB) -- KEEP BOTH (a real
  low/high split). Which LTX, Wan i2v/ti2v, image model; "visualizer = audio-reactive, no scene
  image".
- STAMP RECIPE IN LEDGER: per-beat `delivered_engine` (video) + `image_engine` + `recipe/quant`,
  DURABLE through `production_ledger._merge_with_disk` (which today drops top-level `images`), so
  every episode self-documents what made it ("what did I use?" is unanswerable from saved files
  today).
- ANNOUNCER + MUSIC = always a radio-themed still (vintage radio in the scene) -- never a black card /
  abstract / fallback; the on-brand bookend default. Image-prompt change
  (`otr_meta_brief_image_prompt.py`).
- ADD `visualizer_rainbow` (CREATIVE, schedule LAST): a GLSL/shader audio-reactive visual (rainbow
  palette + plasma / flow-fields / bloom / feedback), reusing `eng_visualizer.py`'s audio analysis
  (FFT / RMS / onsets) to drive shader uniforms. Own `/roundtable` for the shader stack + creative
  direction BEFORE building. Register + CAPABILITIES row + node-87 dropdown + label.

---

## BUG-411  flux CINEMATIC RESTORE  [look-QA, parallel track]

ROOT CAUSE: the 6/5 flux pipeline rewrite into `_otr_image_engines/flux_gen1.py` +
`otr_meta_brief_image_prompt.py` DROPPED the look levers (model/steps/cfg/sampler are identical).
RESTORE: (1) a FluxGuidance node @ ~3.5 (flux_gen1 has none -- biggest factor), (2) the cinematic
style suffix, (3) the radio broadcast-distress suffix, (4) bookend seed 4242, (5) the portrait style
line. Wire the FluxGuidance node into the JSON if the image path runs through the graph.

---

## OPEN QUESTIONS FOR THE PANEL

1. S-A primary: boomerang-loop-extend (composite side) vs phrase-chunk-the-audio (S-C side) vs render
   more frames -- which is right given the 49-frame 14B cap AND the byte-identical-audio constraint?
2. S-F inject point: cleanest seam to feed a frozen ledger+audio into the still/video stages without
   touching the production graph, given `_merge_with_disk` drops top-level `images`.
3. NO-FALLBACKS blast radius: which call sites assume a fallback exists (composite, resolver, sweep
   harness) and will hard-break when the chains are ripped out? Sequence so the suite stays green.
4. Order: is S-F-first correct, or does a correctness fix (S-A) need to land before the fixture is
   trustworthy as a baseline?

---

## APPENDIX -- PANEL ALSO CHECK THESE ComfyUI CUSTOM-NODE INVARIANTS

(Cite the real node file/class for every claim; if you cannot see the code, write "verify: <what>".)

1. NODE-CLASS CONTRACT: every exported node class is in `NODE_CLASS_MAPPINGS` (+ a
   `NODE_DISPLAY_NAME_MAPPINGS` label) -- a defined-but-unmapped class is dead. `INPUT_TYPES` is a
   @classmethod dict (`required`/`optional`/`hidden`); `RETURN_TYPES` is a tuple (trailing comma for
   one output) length-matched to `FUNCTION`'s return; `CATEGORY`/`FUNCTION` set. Widget order is
   POSITIONAL -- appending an optional input is safe; inserting mid-list silently shifts saved widget
   values (this is why retiring engines / adding `visualizer_rainbow` must touch the JSON in the same
   change).
2. TENSOR LAYOUT: IMAGE = float32 [0,1], [B,H,W,C] (channels LAST); MASK = [B,H,W]; LATENT =
   {"samples": tensor}. Flag channels-first / missing-batch-dim / hard-coded device assumptions.
3. VRAM / MODEL MANAGEMENT: heavy models load through `comfy.model_management` (residency/offload/
   eviction managed there), not pinned in module globals with no free path. Flag any plan that holds
   a model resident across runs without eviction (ties to the <=14.5 GB ceiling + the inter-beat
   reclaim).
4. IS_CHANGED / CACHING: a node with hidden external state (file / clock / RNG / network) must
   implement `IS_CHANGED` so it does not serve stale cache. Flag the smoke-fixture / ledger-inject
   path for cache correctness (S-F must not serve a stale render).
5. IMPORT ISOLATION: no heavy / optional imports (torch extras, model libs, CUDA ext) at module top
   level -- lazy-import inside the node method; no import-time side effects (weight downloads, file
   opens). Flag any new engine (`visualizer_rainbow` shader stack) that imports a shader lib at top
   level.
