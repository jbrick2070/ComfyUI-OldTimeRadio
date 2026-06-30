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
- **wan_i2v (14B) / wan_ti2v (5B)** -- VRAM peak + real camera motion eyeball (single low-noise expert risk, GO_FORWARD 4A S3). Isolated, reset between.
- **mesh_stage** (x3) -- prove it renders the Blender mesh and does NOT fall back to still_parallax (Blravender exe present); check the histogram shows `mesh_stage`, not the floor.
- **ltx_video** (announcer, other_beats) -- only music ran (PASS); confirm the other 2 slots.

## B2. VISUAL LEGIBILITY FLOOR + IMAGE CONTRACT (operator-directed 2026-06-29) -- HIGH PRIORITY

ROOT CAUSE CONFIRMED (grounded on `signal_lost_weight_of_the_blueprints_20260629_163656`):
the saved ledger has **NO `images` key** and all 3 cast rows have **empty portrait
fields `{}`**, yet `stills/stills_manifest.json` beside it HAS the clean contract
(`c01` portrait, `role: announcer_visual`, real path + content_hash). So the clean
announcer portrait exists but the video render never got a durable "beat -> portrait"
mapping -> HuMo animated a wrong/empty/degraded init -> the murky helmet-blob center
plate. The source still is fine; the **image contract between stills_manifest and the
ledger/render is broken.** This passes the completion gates (obs ships, audio
byte-identical) while shipping a visually broken clip -- same class as the duration-gate
bug. Fix = QUALITY FLOOR, not choice-limiting (keep every engine selectable).

Sequenced (video-only; master audio immutable; reuse the existing humo->still_parallax
LOUD durable-restamp chain; suite + Bug Bible + B7 per chunk):
1. **Preserve/stamp `ledger['images']`** into the final saved ledger (or reload
   `stills_manifest.json` before video render) so each beat KNOWS its portrait/still.
   PRIMARY fix -- likely resolves the murk at the source (HuMo gets the clean c01).
2. **Stamp per-beat `init_source` / `init_image`** into the clip manifest (proves #1).
3. **Legibility guard after each generated clip:** sharpness (variance-of-Laplacian
   RATIO vs the source -- RELATIVE/catastrophic only; HuMo 480x832 is inherently softer,
   an absolute "blurrier than source" check would flag every HuMo beat), motion (reuse
   freezedetect), subject-presence (face-detect = phase 2, heavier).
4. **On guard failure, composite the clear source still with subtle parallax/pan** --
   never ship a murky/dead generated clip when a clean source plate exists.
5. **Record `attempted_engine` + `delivered_engine` + `fallback_reason`** in the ledger
   (the existing A2 restamp pattern); the dropdown choice is preserved as "attempted".

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
