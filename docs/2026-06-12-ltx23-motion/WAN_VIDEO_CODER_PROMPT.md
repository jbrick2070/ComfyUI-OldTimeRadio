# CODER-WINDOW KICKOFF -- Wan 2.2 video engines (I2V-14B + TI2V-5B), smoke first

Paste as message #1 of a fresh CODER window. Goal: prove TWO selectable Wan 2.2
VIDEO engines in the FAST smoke harness -- one b-roll motion clip from each for
Jeffrey's eyeball -- BEFORE any episode wiring. **Lip-sync stays SEPARATE** on the
existing LatentSync/HuMo engines (talking beats route there); the Wan engines do
b-roll + camera motion only. Supersedes the WAN_S2V prompt (cleaner separation).

> **Hardened 2026-06-12 (planner).** The earlier draft called the 16GB engine a
> "verify/enable, not a build." That was WRONG -- see the grounded reality below.
> Read TASK 0 and the DECISION GATE before you touch code.

## The two engines (one family, two sizes -> OTR's 8gb/16gb profile tiers)
- **16GB tier: Wan 2.2 I2V-A14B.** The motion/quality leader at 16GB. **But the
  full A14B is a TWO-EXPERT MoE (high-noise + low-noise, sigma-split sampling).
  Only the LOW-noise expert is on disk** (see grounded facts). See the DECISION
  GATE -- we smoke the low-noise-only single-sampler graph FIRST (it is what is
  wired and needs zero download); the full two-expert A14B is an explicit upgrade
  if the low-noise motion is not enough.
- **8GB tier: Wan 2.2 TI2V-5B.** Dense 5B (NOT a MoE), does text- AND
  image-to-video, ~4-6GB GGUF. The small-tier / distribution pick. **Needs the new
  Wan2.2 high-compression VAE -- which is NOT on disk** (the I2V-14B uses the older
  wan_2.1_vae). So the 8GB tier is TWO fetches: the 5B model AND the wan2.2 VAE.
- Optional on the 16GB tier: a **Wan camera LoRA** (dolly/pan/orbit). Verify it
  exists for Wan 2.2 I2V before relying on it.

## Grounded facts (verified on disk 2026-06-12 -- supersedes the old "mostly here")
- **ALL models live under `C:\ComfyUI-Models`** (operator, canonical). Diffusion
  weights -> `C:\ComfyUI-Models\diffusion_models\`; text encoders ->
  `\text_encoders\`; VAEs -> `\vae\`. The headless launcher's
  `_otr_headless_model_paths.yaml` maps this tree.
- **ON DISK (confirmed):**
  - `diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` (~13.3GB) --
    **LOW-noise expert ONLY.**
  - `text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors` (the TE the engine
    expects; `CLIPLoader type="wan"`).
  - `vae\wan_2.1_vae.safetensors` (correct VAE for I2V-14B).
- **NOT on disk (each is a fetch + sha256 + license record, fail-closed, no
  runtime download):**
  - The Wan 2.2 **HIGH-noise 14B expert** (needed for true two-expert A14B).
  - **Wan 2.2 TI2V-5B** (any quant).
  - The **Wan2.2 VAE** (the 5B model's high-compression VAE; wan_2.1_vae will NOT
    drive the 5B correctly).
  - Any Wan camera LoRA.
- GGUF loader `UnetLoaderGGUF` is INSTALLED. Wan S2V sampler nodes are NOT
  installed (TI2V/I2V use `WanImageToVideo`-class nodes -- confirm in TASK 0).
- **`nodes/_otr_video_engines/eng_wan_i2v.py` EXISTS but its graph is a PLACEHOLDER.**
  Its docstring promises "two-expert HIGH/LOW MoE + ordered LoRAs + ModelSamplingSD3
  sigma-split," but the actual `_build_graph` is a SINGLE `UNETLoader` + SINGLE
  `KSampler` (`uni_pc`/`simple`, 20 steps, cfg 3.5) explicitly marked
  "ASSUMED native ... VERIFY-ON-GPU." So as-wired it runs **low-noise-expert-only,
  single-pass** -- NOT the full A14B the marketing line implies. It DOES already
  do the right VRAM thing (`free_after_use=True, keep={unet,vae,terminal}`) so the
  umt5 TE is freed before the sampler -- but that is UNVERIFIED on GPU.

## TASK 0 -- verify node signatures + isolation BEFORE any render (do not skip)
The exact failure that burned a whole leg THIS session was node-API drift
(`ltx_orbit` passed a `positive=` kwarg the rewritten node no longer accepts).
Do not repeat it.
- Confirm the installed signatures of `UNETLoader`, `CLIPLoader` (does it accept
  `type="wan"` + `device`?), `WanImageToVideo` (inputs: width/height/length/
  batch_size/positive/negative/vae/start_image?), `KSampler`, `VAEDecode`, and for
  the 5B the TI2V node class + its VAE/TE. Reconcile against
  `eng_wan_i2v._node_candidates()` / `_build_graph()` and FIX the graph to match
  the installed nodes before you render anything.
- **SageAttention / BUG-070:** sage is currently IMPORTABLE on this box (baseline
  is 4136/5, NOT 4141/0 -- the 5 sage/dep tests are a known drift, not your
  regression; do not chase them). `WanI2VEngine.resolve_isolation()` ESCALATES wan
  to a **cu128 sidecar** when SageAttention is resident. The sidecar is likely not
  provisioned -> the smoke could silently divert or fail. DECIDE + state which:
  (a) uninstall/disable sage for the smoke so wan runs in-process (cleanest, matches
  intent), or (b) confirm the sidecar route actually works. Record the choice.

## DECISION GATE -- 16GB tier scope (state your choice, then proceed)
- **Path A (recommended, fast): low-noise-expert-only smoke.** Zero download. Fix
  the placeholder graph per TASK 0, enable, smoke one clip. If the motion is good
  enough for Jeffrey's eye, the 16GB tier is DONE for now.
- **Path B (upgrade, only if A's motion is weak): true two-expert A14B.** Fetch the
  HIGH-noise expert, rebuild the graph with TWO `UNETLoader`s + `ModelSamplingSD3`
  sigma-split + `KSamplerAdvanced` start/end-step handoff (high-noise early steps ->
  low-noise late steps). Heavier, but the real A14B motion. Sequence A before B; do
  not pre-fetch B's 13GB unless A disappoints.

## Tasks (in order)
1. **TASK 0 (above) first.** Then **16GB tier -- Path A smoke (no download).** Boot
   the canonical headless server with `OTR_ENABLE_WAN_I2V=1` and
   `OTR_WAN_I2V_CKPT=C:\ComfyUI-Models\diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`.
   Build `scripts/otr_wan_smoke.py` on the `scripts/otr_ltx_motion_smoke.py` pattern
   (still in -> short clip out via /prompt, SaveWEBM, optional MAD via
   `scripts/otr_ltx_mad.py`): render ONE b-roll motion clip (radio-console still +
   a motion prompt) at the engine default 832x480, fixed seed.
   - **ASSERT wan_i2v is the engine that ran** (final_engine in the trace). If it
     fell back to still_kenburns/other, FAIL LOUD -- do NOT pass a fallback clip
     (the CS-1 latentsync "fallback-PASS" trap).
   - **Record BOTH numbers:** whole-run peak NVML AND render-phase peak NVML
     (separately -- the V-3 gate vs the staged-stills confusion is CS-2; the
     render-phase peak is the one that must be <=14.5GB). Also render time + seed +
     MAD (MAD is secondary -- it oversold the LTX warp; the gate is visual).
   - If the fp8 14B render-phase peak busts 14.5GB, fetch the **GGUF Q5_K_M**
     (~10-11GB) and load via `UnetLoaderGGUF` instead.
   - **Determinism:** render the clip twice with the same seed; assert the outputs
     match (V-7) or log why they cannot.
2. **8GB tier -- fetch + wire TI2V-5B (TWO fetches).** Pull **Wan 2.2 TI2V-5B GGUF
   Q6** (or Q5_K_M) AND the **Wan2.2 VAE** into `C:\ComfyUI-Models\`; record HF repo
   + sha256 + license for each; fail-closed if absent (no runtime download). VERIFY
   the operator's ~4-6GB model-size claim against the actual file. Wire it as a
   SECOND selectable Wan engine (TI2V is dense, not a MoE; uses a different node +
   the wan2.2 VAE + likely the same umt5 TE -- confirm the exact node class).
   Smoke ONE b-roll clip, same asserts as task 1 (engine-in-trace, both NVML peaks,
   determinism).
3. **Eyeball gate:** present BOTH webms to Jeffrey (I2V-14B vs TI2V-5B, same still +
   prompt) under `docs/2026-06-12-ltx23-motion/wan_clips/`. Bar is VISUAL (real
   camera motion, still preserved, no warp), NOT MAD alone. Lock nothing until he
   confirms.
4. **Only after eyeball PASS:** map the two engines onto the switchable 8gb/16gb
   profile tiers (I2V-14B = 16gb video engine, TI2V-5B = 8gb video engine), keeping
   lip-sync on LatentSync/HuMo for talking beats. **Forward-link CS-3:** the episode
   wiring (per-beat role routing: announcer/character talking -> lip-sync engines;
   b-roll/console -> Wan) is a SEPARATE step, and CS-3 is the open risk there (Wan +
   HuMo co-staging in one episode may bust 16GB). Record each engine's standalone
   render-phase peak now so that decision has real numbers.

## Hard rules (unchanged)
- Single resident heavy <=14.5GB (host NVML, render-phase). 100% local after the
  TI2V + VAE (+ optional high-noise) fetches. Frozen audio spine untouched (the Wan
  video engines are SILENT motion; audio mux stays byte-identical;
  `test_audio_byte_identical` green). Determinism (seed-keyed). UTF-8 no BOM, SFW.
  Commit per green chunk, do NOT push unprompted (operator gate).
- Run full tests/ + Bug Bible after any code change. Baseline is **4136/5** on this
  box (sage drift), not 4141/0 -- do not treat the 5 sage tests as your regression.
- Use the canonical launcher (`scripts/_otr_soak_server_launch.cmd`) + the auto
  render-launcher + watchdog (`scripts/otr_run_leg.ps1`). Aggressively reset the GPU
  before EVERY boot (kill python, confirm :8000 free, confirm NVML at desktop
  baseline) -- the soak harness leaves the server RESIDENT by design.
- COORDINATION: ONE coder window in the repo code at a time; claim/serialize via the
  GO file. Update GO_FORWARD_PLAN.md + the otr-build-tracker every session.

## ComfyUI quirks & likely bugs (planner QA pass 2026-06-12 -- read before coding)
Grounded in `wrapper_bridge.run_graph`, `eng_wan_i2v.py`, and
`scripts/otr_video_dep_pilot.py`.
- **USE CORE Comfy Wan nodes, NOT the KJ wrapper.** The dep-pilot's `assumed_call`
  for wan is the KJ "two-expert MoE + lightx2v/SVI/distill LoRAs + ModelSamplingSD3"
  wrapper, whose **gate-F pin audit wants numpy<2 / transformers<=4.51.3**. This box
  is numpy 2.4 / transformers 5.5 -> the KJ wrapper would FAIL or contaminate the
  venv, and KJNodes is what pulls SageAttention (BUG-070 -> the sidecar escalation).
  The placeholder graph already uses CORE nodes (`UNETLoader`/`CLIPLoader`/
  `WanImageToVideo`/`KSampler`/`VAEDecode`) -- KEEP IT THAT WAY. Build Path B's
  two-expert split with CORE nodes too (two `UNETLoader`s + `ModelSamplingSD3` +
  `KSamplerAdvanced` start/end-step handoff). Core Comfy supports A14B natively; you
  do NOT need the KJ wrapper. This is also why staying in-process is safe.
- **`ModelSamplingSD3` is MISSING from the placeholder graph.** Wan 2.2 wants a
  sigma shift (~8.0 for 14B, ~5.0 for 5B) even for a single expert; without it
  motion/quality is off. Add `ModelSamplingSD3` between `UNETLoader` and the
  sampler.
- **`free_after_use` is a ref-drop + soft reclaim, NOT a guaranteed detach.** In
  `run_graph` the `clip` (umt5) output is dropped after `neg` runs (before the
  sampler) and `_soft_free()` fires -- but on this box's DynamicVRAM the soft free
  may detach ZERO (the `ltx_orbit`/e9743cc precedent: it reported "detached 0").
  The REAL eviction happens when `KSampler` calls `load_models_gpu` under VRAM
  pressure. So the **render-phase NVML peak is the only truth** for whether umt5
  (~5.2GB) actually leaves before the 13.3GB UNET loads (this is the CS-4 mechanism
  -- it killed the HuMo 14B). Measure it; do not trust the soft free.
- **The isolated smoke SIDESTEPS CS-2** (no Flux/portrait stills are staged), so its
  render-phase peak is the TRUE standalone engine cost -- cleaner than a full-episode
  leg where staged stills inflate NVML. Good for the ceiling decision; bad as a proxy
  for the eventual co-staged episode (that's CS-3).
- **TI2V-5B likely needs a DIFFERENT latent node.** Core Comfy split out
  `Wan22ImageToVideoLatent` for the 5B's high-compression VAE -- `WanImageToVideo`
  (the 2.1/2.2-14B node) may not drive the 5B correctly. Confirm the exact node +
  that the 5B uses the wan2.2 VAE (not wan_2.1_vae) and (probably) the same umt5 TE.
- **GGUF branch needs a loader swap.** `_node_candidates()` lists `UNETLoader` only;
  the ceiling-bust GGUF fallback must swap in `UnetLoaderGGUF` (installed) for the
  `unet` node. The TI2V-5B GGUF needs it too.
- **clip_vision:** Wan 2.2 I2V-A14B dropped the clip_vision encoder that 2.1 used.
  Core `WanImageToVideo` exposes `clip_vision_output` as OPTIONAL -- leave it
  unset for 2.2 (the placeholder correctly omits it). Confirm on the installed node.
- **`length` must be 4n+1** (`quantize_frames_4n1`, Wan-VAE 4-frame compression) --
  already handled; keep it when you rebuild.
- **fp8_scaled load dtype:** `UNETLoader weight_dtype="default"` is right for the
  `_fp8_scaled` file; do not force `fp8_e4m3fn_fast` unless a black-frame decode
  forces it.

## Why (decided 2026-06-12, roundtable + operator)
LTX-2.3 22B = real motion but won't fit 14.5GB (panel-unanimous). v0.9 2B fits but
warps (operator eyeball). Wan 2.2 is the livelier family that fits BOTH tiers and
keeps lip-sync separate, so neither tier carries a 14B model just for lips. One
family, two sizes, drops onto the profile tiers -- with the grounded caveat that the
16GB engine is currently a low-noise-only placeholder and the 8GB tier is two
unfetched files.
