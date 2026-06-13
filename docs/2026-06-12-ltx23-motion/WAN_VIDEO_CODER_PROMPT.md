# CODER-WINDOW KICKOFF -- Wan 2.2 video engines (I2V-14B + TI2V-5B), smoke first

Paste as message #1 of a fresh CODER window. Goal: prove TWO selectable Wan 2.2
VIDEO engines in the FAST smoke harness -- one b-roll motion clip from each for
Jeffrey's eyeball -- BEFORE any episode wiring. **Lip-sync stays SEPARATE** on the
existing LatentSync/HuMo engines (talking beats route there); the Wan engines do
b-roll + camera motion only. Supersedes the WAN_S2V prompt (cleaner separation).

## The two engines (one family, two sizes -> OTR's 8gb/16gb profile tiers)
- **16GB tier: Wan 2.2 I2V-A14B.** The current motion/quality leader at 16GB:
  real camera paths, weighty motion, less "AI float." Stages alone under 14.5GB.
- **8GB tier: Wan 2.2 TI2V-5B.** Best small i2v of 2026; ~4-6GB GGUF, comfortable on
  8GB with offload; does text- AND image-to-video. The small-tier / distribution pick.
- Optional on the 16GB tier: a **Wan camera LoRA** (dolly/pan/orbit) for explicit
  camera control on the console b-roll shots (verify it exists for Wan 2.2 I2V).

## Grounded facts (verified on this box 2026-06-12) -- the 14B is MOSTLY HERE
- **ALL models live under `C:\ComfyUI-Models`** (operator, canonical). GGUFs ->
  `C:\ComfyUI-Models\diffusion_models\`. The headless launcher's
  `_otr_headless_model_paths.yaml` already maps this tree.
- **OTR ALREADY HAS `nodes/_otr_video_engines/eng_wan_i2v.py`** -- a `WanI2VEngine`
  (MotionEngineBase) that builds a `WanImageToVideo` graph, gated by
  `OTR_ENABLE_WAN_I2V` + `OTR_WAN_I2V_CKPT`. So the 16GB I2V-14B engine is a
  VERIFY/ENABLE, not a build.
- **On disk: `C:\ComfyUI-Models\diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`
  (13.3GB)** + `vae\wan_2.1_vae.safetensors`. The 16GB I2V-14B model is present (fp8).
- GGUF loader **`UnetLoaderGGUF` is INSTALLED**.
- **NOT on disk: Wan 2.2 TI2V-5B** (any quant) -> one fetch needed.

## Tasks (in order)
1. **16GB tier -- ENABLE + SMOKE the existing engine first (no download).** Boot the
   canonical headless server with `OTR_ENABLE_WAN_I2V=1` and
   `OTR_WAN_I2V_CKPT=C:\ComfyUI-Models\diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors`.
   Build `scripts/otr_wan_smoke.py` on the `scripts/otr_ltx_motion_smoke.py` pattern
   (still in -> short clip out via /prompt, SaveWEBM, MAD via `scripts/otr_ltx_mad.py`):
   render ONE b-roll motion clip (radio-console still + a motion prompt). Record
   render time + **peak NVML (must stay <=14.5GB)** + MAD. If the fp8 14B busts the
   ceiling, fetch the **GGUF Q5_K_M** (~10-11GB) instead and load via UnetLoaderGGUF.
2. **8GB tier -- fetch + wire TI2V-5B.** Pull **Wan 2.2 TI2V-5B GGUF Q6** (or Q5_K_M)
   into `C:\ComfyUI-Models\diffusion_models\`; record HF repo + sha256 + license,
   fail-closed if absent (no runtime download). VERIFY the operator's ~4-6GB size
   claim against the actual file. Wire it as a SECOND selectable Wan engine
   (clone the eng_wan_i2v pattern, or a TI2V variant -- TI2V uses a different node /
   does t2v+i2v; confirm the exact node class + the VAE/text-encoder it needs).
   Smoke ONE b-roll clip from it too.
3. **Eyeball gate:** present BOTH webms to Jeffrey (I2V-14B vs TI2V-5B, same still +
   prompt). Bar is VISUAL (real camera motion, still preserved, no warp), NOT MAD
   alone (MAD oversold the LTX warp this session). Lock nothing until he confirms.
4. **Only after eyeball PASS:** map the two engines onto the switchable 8gb/16gb
   profile tiers (I2V-14B = 16gb video engine, TI2V-5B = 8gb video engine), keeping
   lip-sync on LatentSync/HuMo for talking beats. The episode wiring (per-beat role
   routing: announcer/character talking -> lip-sync engines; b-roll/console ->
   Wan) is a SEPARATE step after the clips look right.

## Hard rules (unchanged)
- Single resident heavy <=14.5GB (host NVML). 100% local after the one TI2V fetch.
  Frozen audio spine untouched (the Wan video engines are SILENT motion; audio mux
  stays byte-identical; `test_audio_byte_identical` green). Determinism (seed-keyed).
  UTF-8 no BOM, SFW. Commit per green chunk, do NOT push unprompted (operator gate).
- Run full tests/ + Bug Bible after any code change. Use the canonical launcher
  (`scripts/_otr_soak_server_launch.cmd`) + the auto render-launcher + watchdog
  (`scripts/otr_run_leg.ps1`).

## Why (decided 2026-06-12, roundtable + operator)
LTX-2.3 22B = real motion but won't fit 14.5GB (panel-unanimous). v0.9 2B fits but
warps (operator eyeball). Wan 2.2 is the livelier family that fits BOTH tiers as
GGUF, the 14B is already wired + on disk, and keeping lip-sync separate means
neither tier carries a 14B model just for lips. One family, two sizes, drops onto
the profile tiers.
