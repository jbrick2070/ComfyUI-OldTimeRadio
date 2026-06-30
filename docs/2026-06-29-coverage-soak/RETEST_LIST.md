# OTR Coverage Soak -- Retest / Retry List (post-soak action plan)

Living doc. Built during the 2026-06-29 30-word coverage soak (every video +
image engine x 3 slots, writer = local gemma, visualizer excluded). The
"confirm from final matrix" rows get filled in when the soak completes
(`scripts/_otr_coverage_matrix.json` + the `otr-coverage-soak` dashboard).

Invariants for every retest: single resident heavy <= 14.5 GB (host NVML);
seed-keyed determinism; LOUD fallbacks; master audio byte-identical; UTF-8 no BOM;
selective CIM reset before each isolated run (never a blanket python kill).

---

## A. HARD-FAILS -- fix at root, then retest in isolation

| Engine / area | Slots | Status | Root cause (grounded) | Retest after fix |
|---|---|---|---|---|
| `ltx_audio_in` | announcer (FAILED), music, other_beats | SOAK_FAIL: `VRAM 15936 MB > 14500` (eng_ltx_av.py:687) | Regression: `7bbce1d8` "bakeoff-winner quality upgrade (PROVISIONAL)" + `fd9edc28` switched to dev-Q3_K_M + **SHARP LoRA** (~15.5 GB). Last-good = `c4d7815b` base recipe, 512x288 = 13688 MB. | Re-fit via recipe/quant (`OTR_LTX_AV_RECIPE` / `distilled_native` / lighter quant), then run each of the 3 slots **in isolation** (selective reset between) and confirm peak <= 14.5 + a real render. |
| _(more TBD)_ | | confirm from final matrix | heavy legs (humo_14B_169 x3, wan x5, mesh_stage x3) not yet run | |

## B. RETEST IN ISOLATION even if they PASS (borderline / eyeball)

- **humo_14B_169** (x3 slots) -- 49-frame cap + mirror-extend (`eng_humo.py:61`); verify motion/reactivity + VRAM headroom. Ties to the operator "not much movement" note. Isolated run + freeze-detect motion probe.
- **HuMo PORTRAIT QUALITY (operator eyeball 2026-06-29, "Weight of the Blueprints" announcer):** the rendered face is MURKY/blurry -- an indistinct gold blob, not a clear announcer. Diagnose source: is the INPUT portrait (flux_gen1 still under `episodes/<ep>/stills/c01_*.png`) already poor, or is HuMo + the upscale/pillarbox degrading it? Inspect the source still vs the HuMo output side by side. Applies to humo / humo_1.7B / humo_14B_169.
- **Title-reveal glitch** -- "WEIGHT OF THE BLU5NV#03W" seen at ~0:00:02 is the intentional title scramble-in (BUG-409 `_title_reveal_progress`, resolves in first ~40%); CONFIRM it resolves to the clean title by ~40% of the window, not a stuck scramble.
- **humo / humo_1.7B / humo_1.7B_169** (x3 each) -- confirm each variant renders its own beats (no silent degrade to the floor) + VRAM.
- **HuMo VARIANT OPTIMIZATION (operator 2026-06-29: "some HuMos may be good, some bad, we never optimized them").** The 4 variants (humo 14B / humo_1.7B / humo_1.7B_169 16:9 / humo_14B_169 16:9) differ in quality/motion/aspect and have NEVER been tuned. BUT the clip-underrun held-frame (S-A) MASKS their true quality -- you only see a frozen tail. So evaluate + optimize them on FILLED clips, i.e. AFTER the S-A clip-fill ships, in a dedicated HuMo-variant pass: per-variant cfg / steps / frame-budget / aspect, eyeball good-vs-bad, set the best default per role. Do NOT judge HuMo variants on the current frozen-tail renders. Ties to S-C HuMo phrase-chunking (shorter beats fit the 177f budget).
  - **REGRESSION FOUND (git): the 1.7B blur is a cfg drop.** Proven sharp config `06e50304`
    (2026-06-07) = `uni_pc / cfg 5.0 / 20-step / no-LoRA`. `fdb93286` (2026-06-17)
    *"de-blue humo_1.7B (cfg 5.0->1.0 kills the blue cast)"* dropped cfg 5.0 -> 1.0;
    current `eng_humo.py:152 _cfg()` defaults to 1.0. The no-LoRA 1.7B is non-distilled
    and needs cfg ~5.0 for guidance -- at cfg 1.0 it UNDERGUIDES -> blur (the 14B distill
    correctly uses cfg 1.0; the change wrongly applied it to the 1.7B too). ENV-TESTABLE
    NOW (no code): re-render a 1.7B beat with `OTR_HUMO_CFG=5.0` (+ `OTR_HUMO_STEPS=20`,
    sampler uni_pc) and A/B vs the cfg-1.0 blur; if sharp, bake cfg 5.0 back as the 1.7B
    default + solve the blue cast separately (color-correct / VAE, not by killing cfg).
- **wan_i2v (14B) / wan_ti2v (5B)** -- VRAM peak + real camera motion eyeball (single low-noise expert risk, GO_FORWARD 4A S3). Isolated, reset between.
- **mesh_stage** (x3) -- prove it renders the Blender mesh and does NOT fall back to still_parallax (Blravender exe present); check the histogram shows `mesh_stage`, not the floor.
- **ltx_video** (announcer, other_beats) -- only music ran (PASS); confirm the other 2 slots.

## B2. VISUAL LEGIBILITY FLOOR + IMAGE CONTRACT (operator-directed 2026-06-29) -- HIGH PRIORITY

ROOT CAUSE CONFIRMED + REPRODUCED (render logs for `weight_of_the_blueprints_163656`
AND `steel_against_skin_170522`): this is a "bad/short generated clip was allowed to
ship" bug, NOT a routing-choice bug and NOT (primarily) a missing-image bug.
- The announcer portraits ARE present at render: `[portrait_ledger] still_b001/b005 ...
  recorded via ledger['images']`. HuMo got real image data.
- `humo_1.7B` UNDERRUNS: `CLIP UNDERRUN (LOUD): shot_b005 rendered 177 frame(s) for a
  434-frame target (41%) -- the composite will HOLD the last frame for the rest of the
  beat ... investigate 'humo_1.7B'`. The held static last-frame IS the murky/dead plate
  (177 = HuMo per-clip frame ceiling vs the long 405-434f announcer beats).
- Completion gates (obs ships, audio byte-identical) PASS regardless -- same class as the
  duration-gate bug: a visually broken clip passes the non-visual gates.
- The saved ledger LOSING `images` (`production_ledger._merge_with_disk` drops top-level
  `images`; in-memory ledger from `OTR_ImageGenDispatcher` DID carry them into
  `OTR_VideoRenderBatch`) is a FORENSIC gap, not the render cause. `video_readiness=0`
  is a stale EARLY check of legacy cast-row portrait fields -- not the modern contract.
Fix = QUALITY FLOOR, not choice-limiting (keep every engine selectable).

Sequenced (video-only; master audio immutable; reuse the existing humo->still_parallax
LOUD durable-restamp chain; suite + Bug Bible + B7 per chunk):
1. **Clip-fill (PRIMARY):** a motion engine that underruns its frame target LOOPS /
   ping-pong-extends to the target (the composite's own recommendation) instead of holding
   the last frame -> motion across the whole beat, no frozen plate.
2. **Legibility guard after each clip:** sharpness (variance-of-Laplacian RATIO vs the
   source -- RELATIVE/catastrophic only; HuMo 480x832 is inherently softer, an absolute
   "blurrier than source" check would flag every HuMo beat), motion (reuse freezedetect),
   subject-presence (face-detect = phase 2, heavier).
3. **On guard failure, composite the clear source still with subtle parallax/pan** --
   never ship a murky/dead/frozen generated clip when a clean source plate exists.
4. **Record `attempted_engine` + `delivered_engine` + `fallback_reason`** in the ledger
   (the existing A2 restamp pattern); the dropdown choice is preserved as "attempted".
5. **SECONDARY/forensic:** preserve `ledger['images']` durably + stamp per-beat
   `init_image_used` / `init_source` (aids diagnosis; not the cause). HuMo phrase-chunking
   (section E) attacks the same underrun root -- shorter beats fit HuMo's 177-frame budget.

## C. SILENT-FALLBACK AUDIT (the "Rendered" column)

- For every PASS leg, confirm the target engine appears in that leg's
  `engine_histogram` (dashboard flags `fallback!` if not). A PASS with the
  target engine ABSENT = silent degrade = treat as a fail-to-fix.
- Confirm from final matrix: list any leg where rendered != requested.

## D. KNOWN BUGS surfaced by the soak -- fix + retest

- **gemma `normalize_length` wrapper-key drift** (EVERY leg, warn-only): model
  returns the plan nested under a top-level `RadioEditPlan` key ->
  `projected_word_total` "missing" -> retry ladder exhausts -> length
  normalization skipped. Fix the LEVER-1 tolerant-unwrap to peel a top-level
  schema-name wrapper; retest on a gemma leg (warning should disappear).
- **obs-final duration gate vs credits tail** -- FIXED + pushed `3991c019`
  (audio-stream == master, video >= audio). Re-confirm green across credits-tail
  episodes in the final matrix (no false SOAK_FAIL on duration).

## E. AUDIO-IN CONDITIONING SPRINT (post-soak, own sprint -- agreed hierarchy)

1. **Observability first** -- per-beat log: recipe, unet/quant, LoRA on/off,
   render canvas, frame length, target frames, audio-source class (clean
   voice / master slice / theme / missing), audio duration, init source,
   explicit phase marker (before-sample / during-sample / during-decode),
   peak VRAM. (Replaces the stale `13688` constant-as-truth.)
2. **Fit `ltx_audio_in` <= 14.5 GB** via recipe/quant/offload -- proven by (1).
3. **Master audio immutable** -- `test_audio_byte_identical` stays green.
4. **`audio_motion_profile`** (rms/peak/onset/silence/brightness/dynamic-range/
   speech-vs-music/duration) shared per beat; drives non-audio engines
   (LTX/Wan/still/3D) via prompt/camera/parallax/light.
5. **HuMo phrase-chunking** for long dialogue instead of mirror-extending 49f.
6. **Probe-gated HQ/reactivity tiers** LAST (resolution only after fit).

## F. ENVIRONMENT / SYSTEM

- Re-run any borderline VRAM engine with other desktop apps CLOSED (clean GPU
  baseline ~1.5 GB) to separate OTR's own budget from external VRAM pressure.
- Confirm `:8000` reset (selective CIM) + GPU at baseline before each isolated retest.

## G. IMAGE ENGINES (5 x 3 image slots = 15)

- `flux_gen1`, `flux2_klein`, `z_image_turbo`, `lumina_image` -- on disk; expect render.
- **`qwen_image`** -- no qwen model seen in `C:\ComfyUI-Models\diffusion_models`;
  expect MISSING_MODEL (loud, named). Decide: install the model + retest, or
  leave as a named gap.
- Confirm each image engine actually minted its still (histogram / stills_manifest),
  not a silent skip.

---

### Final-matrix fill-in (complete when soak done)
- Total legs PASS / FAIL: __ / __
- Hard-fails (engine, slot, reason): __
- Silent fallbacks (requested -> rendered): __
- Audio byte-identical held on all PASS legs: __
