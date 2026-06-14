# OTR overnight Wan window -- 2026-06-14

Single source of truth is `docs/GO_FORWARD_PLAN.md` (section 1). This is the quick read.

## What shipped (HEAD `ed6bc7e`, pushed to `v2.0-alpha`, HEAD==origin)

- **`wan_ti2v` 8GB-tier engine BUILT** -- the deferred Wan2.2 TI2V-5B sibling. Core Comfy nodes:
  `UnetLoaderGGUF` (Q5_K_M) -> `ModelSamplingSD3` (shift 5.0) -> `Wan22ImageToVideoLatent` (the 5B
  latent node) -> KSampler (pos/neg wired DIRECT) -> `VAEDecode` with the **Wan2.2 VAE**. The 5B core
  node class was captured from a **live `/object_info`** before any code was written (VERIFY-AT-BUILD).
- **`wan_shared` mixin** -- the pure dims/aspect/materialize/clip-contract helpers are shared by both
  Wan engines. `wan_i2v` was refactored onto it (behavior-preserving; it is NOT a `WanI2VEngine`
  subclass -- loaders/nodes/graph stay separate, per the plan).
- **M8** (Wan2.2-VAE fail-closed: rejects an empty VAE name or the 2.1 VAE) + **S2** (CAPABILITIES row
  `medium`/8000) -- both LANDED (were deferred).
- **Wiring:** registry register, sweep import, dep-pilot `OPT_IN_ENGINES`, and the WAN launch lane now
  enables both engines. No change to `otr_scifi_16gb_full.json` -- like every other opt-in engine,
  `wan_ti2v` is selectable-not-default and the combo is runtime-populated (the workflow-validator tests
  stay green).
- **Models** fetched to `C:\ComfyUI-Models` (`diffusion_models\Wan2.2-TI2V-5B-Q5_K_M.gguf` +
  `vae\wan2.2_vae.safetensors`) with sha256 + license recorded in `MODEL_MANIFEST.json` (GGUF
  Apache-2.0).
- **Tests:** 22 new unit tests; full suite **4249 pass / 28 skip**, Bug Bible **16 pass**, audio
  byte-identical green. AST-clean, no BOM, HEAD==origin verified.

## Live validation (both engines render on the 5080)

- **wan_i2v** -- a real full-episode `--acceptance --only wan` leg ran **21 clean 14B sampler passes**
  (post mixin-refactor, no VRAM-ceiling-breach assertion fired). The refactor is proven working live.
- **wan_ti2v** -- a bare-graph 5B smoke (the exact engine node graph) rendered **25 frames via the 2.2
  VAE in 35s, ~9 GB peak** (= the lighter 8GB tier). RESULT: PASS. Clip for your eyeball:
  `docs/2026-06-14-wan-ti2v/wan_ti2v_5b_smoke.mp4`.

## YOUR MORNING QUEUE (operator-gated -- I can't approve these)

1. **WEBM EYEBALL:** compare the I2V-14B clip vs the TI2V-5B clip (`wan_ti2v_5b_smoke.mp4`). Bar = real
   camera motion, still preserved, no warp. If the 14B motion is too subtle, the Path B two-expert
   HIGH/LOW handoff is the mitigation (S3), not a knob tweak.
2. **Formal `--acceptance` GREEN exit (optional):** the multi-leg sweep is impractically slow because the
   `music_visual=wan` leg renders the ENTIRE music bed as Wan video (~21 clips/leg, ~20+min/leg). Run it
   attended/selectively, or with a short-music profile for the wan legs. The per-engine validation is
   already done; this is just the formal gate.
3. **M9 CS-3 instrumented proof:** partial evidence in hand (the mixed leg ran without a breach). A clean
   per-beat-peak + reclaim-drain capture remains.

## Overnight soak (your request)

A randomized **120-word episode soak** is running on the default lane (HuMo on; the 16gb_full profile =
LTX announcer/music + humo_1.7B character + flux images + full audio). Up to 8 episodes or 6 hours, OS-
entropy cast/style variety. Verdicts: `scripts/_otr_120word_soak_summary.json` (+ `_otr_120word_soak.log`).
