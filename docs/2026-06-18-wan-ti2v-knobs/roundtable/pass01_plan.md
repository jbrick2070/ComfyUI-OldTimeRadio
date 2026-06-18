# wan_ti2v "most solid low floor" -- hardened plan (roundtable pass01, converged)

GOAL: the most solid, portable, dependable 480p floor for 8GB / Mac (MPS) / AMD.
NOT a quality bake-off. The Lightning LoRA is CUT (it's a speed lever, not a VRAM
lever, and breaks GGUF/core-only portability) -- if it ships at all it's a higher
tier, not the floor.

## The real problem (grounded): the floor does not fit the floor
Current default renders 33 frames @ 832x480 -> ~13.1 GB peak on a 5080 (residency +
video-VAE decode). On a real 8GB card that fail-closes. Step count does NOT change
this. The floor needs VRAM + portability + reliability hardening, in that order.

## Work, in priority order

### P1 -- VRAM (the actual blocker)
- Add a **tiled VAE decode** path: `VAEDecodeTiled` as a `vaedecode` node candidate
  (env-gated, default-on for the floor tier) -- the decode is a top peak driver.
- Lower the frame floor: `_TI2V_MIN_FRAMES` 33 -> **17** (4n+1 preserved) for the
  8GB tier; add `_TI2V_DEFAULT_FRAMES` and use it as the fallback instead of
  misusing `target_fps` (25 Hz) as a frame count.
- Document/require a low-VRAM offload config for the floor (ComfyUI `--lowvram` /
  sequential offload); the engine already frees the encoder before decode.
- **Re-measure peak at `VramPeakProbe(interval_s=0.1)`** on a memory-constrained
  config; target measured peak < 8GB before claiming floor-fit. (My 0.7s sample of
  13.1 GB may understate the true peak.)

### P2 -- Portability default (GGUF-vs-fp8 conflict RESOLVED by grounding)
**GROUNDED FACT (2026-06-18 search):** Float8 is NOT supported on the Mac MPS
backend -- the official fp8 `umt5_xxl_fp8_e4m3fn_scaled.safetensors` throws a
`Float8_e4m3fn` TypeError on Apple Silicon (ComfyUI issue #9255). So the CURRENT
default CLIP is Mac-broken, and "fp8 is more portable" is FALSE. GGUF dequantizes to
fp16 at load -> sidesteps fp8 -> is the low-VRAM cross-platform path (Wan GGUF runs
6-12 GB). Resolved method for the floor:
- **UNET:** keep **GGUF** (`UnetLoaderGGUF`, 5B Q5_K_M ~3.6 GB, dequant->fp16,
  cross-platform torch). fp16 safetensors UNET = the bigger Mac-safe fallback if a
  backend can't run ComfyUI-GGUF.
- **CLIP (umt5):** MOVE OFF fp8 -> a **GGUF umt5** encoder
  (`umt5-xxl-encoder-Q5_K_M.gguf`, low-VRAM, Mac-safe) or an **fp16 umt5**
  safetensors (simpler, bigger). This is a CHANGE from the current fp8 default and
  is required for Mac. (`CLIPLoaderGGUF` is a ComfyUI-GGUF node -- same dep family
  as `UnetLoaderGGUF`, already required.)
- **VAE:** wan2.2 vae fp16/bf16 + `VAEDecodeTiled`.
- **Sampler:** core **`euler`** + `simple` (rock-solid cross-platform). Add an
  `OTR_WAN_TI2V_SAMPLER` **whitelist validated in `assert_usable`** (fail-closed on
  a non-portable value) -- drop `uni_pc`/`sa_solver`/`MoEKSampler` from the floor.
- shift: test 3.0 vs 5.0; pick the steadier.
- **STILL VERIFY-AT-BUILD on a real Mac/AMD:** confirm ComfyUI-GGUF dequant runs on
  MPS/ROCm/DirectX (it is pure-torch, expected yes) + a real 8GB peak measurement.

### P3 -- Reliability hardening
- CFG/steps coupling guard: if steps are low + a distill is ever configured, force/
  validate cfg~1.0 (prevents the silent "well-formed garbage" path).
- OOM: a pre-flight frame/size estimate or a catch+retry-at-lower-length, so the
  floor degrades predictably instead of rendering-then-asserting.
- License guard in `assert_usable` IF a LoRA path is ever added (Apache/MIT only).

## A/B (after P1-P2 land) -- a FIT test, not a beauty contest
- **A** = current baseline (33f, uni_pc, shift 5, GGUF) -- expected to OOM 8GB.
- **E (the hardened floor)** = euler, shift 3.0, 17 frames, VAEDecodeTiled, lowvram.
Same still + seed; score: does it FIT (<8GB measured) + render cleanly + hold the
still. Winner becomes the floor-tier default (env-gated; bigger cards can override
up). Operator eyeballs the clip.

## Out of scope (explicit)
720p, higher-step quality tiers, audio-in, the Lightning/distill LoRA -> all belong
to the LTX audio-in / higher tier, NOT this accessible floor.
