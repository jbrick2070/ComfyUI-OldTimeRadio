# wan_ti2v "most solid low floor" -- FINAL hardened method (converged pass02, build-ready)

The stable, low-VRAM, system-agnostic (8GB / Mac MPS / AMD) Wan2.2 TI2V-5B floor.
The Lightning/distill LoRA is OUT (speed lever, not a VRAM lever; GGUF+LoRA needs a
custom loader; that's a higher tier). Touch `eng_wan_ti2v.py` only (+ tests + doc).

## The method (what the floor IS)
- **UNET:** GGUF `Wan2.2-TI2V-5B-Q5_K_M.gguf` via `UnetLoaderGGUF` (dequant->fp16;
  cross-platform torch; avoids fp8). fp16 safetensors UNET = bigger Mac-safe fallback.
- **CLIP (umt5):** **OFF fp8** -- floor default GGUF `umt5-xxl-encoder-Q5_K_M.gguf`
  via `CLIPLoaderGGUF` (fp8 throws Float8_e4m3fn TypeError on MPS -- ComfyUI #9255).
  fp16 umt5 safetensors = the `CLIPLoader` fallback.
- **VAE:** Wan2.2 vae + **`VAEDecodeTiled`** (tile ~256 / overlap) on the floor.
- **Sampler/sched:** core **`euler` / `simple`** only (rock-solid cross-platform).
- **Frames:** 17 (4n+1) floor. **Steps/cfg/shift:** keep 30 / 5.0 / 5.0 for now
  (steps don't drive VRAM; a cheap shift 3.0-vs-5.0 eyeball is optional, non-gating).
- **Offload:** operator runs the ComfyUI server with `--lowvram` for the 8GB tier
  (DOC -- the in-process engine cannot set it per-render).

## Build checklist (unanimous, grounded vs eng_wan_ti2v.py)
1. **CLIP loader:** add `_clip_loader_mode()` (mirror `_loader_mode()`); make
   `_node_candidates["clip"]` conditional `("CLIPLoaderGGUF",)` vs `("CLIPLoader",)`;
   default basename -> the GGUF umt5; reconcile the clip inputs dict per loader; new
   env `OTR_WAN_TI2V_CLIP_LOADER` (gguf|safetensors, default gguf) +
   `OTR_WAN_TI2V_CLIP_NAME`. Add `CLIPLoaderGGUF` availability to `assert_usable`
   (named "install ComfyUI-GGUF" error).
2. **Tiled VAE:** env `OTR_WAN_TI2V_TILED_VAE` (default 1) -> `_node_candidates
   ["vaedecode"]` = `("VAEDecodeTiled",)` else `("VAEDecode",)`; add tile_size/overlap
   to the `vaedecode` inputs in `_build_graph` when tiled; floor fails-closed if absent.
3. **Frames:** `_TI2V_MIN_FRAMES` 33->17; add `_TI2V_DEFAULT_FRAMES=17`; in
   `render_clip` change `or self.target_fps` -> `or _TI2V_DEFAULT_FRAMES`; clamp the
   floor tier so an upstream `target_frame_count` can't push 33+ without an explicit
   higher-tier override.
4. **Sampler whitelist:** `_build_graph` default `uni_pc`->`euler`; `_PORTABLE_SAMPLERS
   = frozenset({"euler"})`; validate `OTR_WAN_TI2V_SAMPLER` in `assert_usable` reading
   the SAME default (unset must pass); fail-closed on non-whitelisted.
5. **Config resolver:** one helper parses+range-checks steps/cfg/shift/sampler/
   scheduler, used by BOTH `assert_usable` and `_build_graph` (no raw int()/float()
   crash mid-render).
6. **VAE guard:** `assert_usable` fail-closed unless the resolved VAE basename is the
   approved Wan2.2 name (not just "not empty / not 2.1").
7. **Probe:** `VramPeakProbe(interval_s=1.0)` -> `0.1` on the measurement path.

## CUT (do NOT build)
LoRA wiring / Lightning / 6-step distill / CFG-distill guard / LoRA license guard
(no LoRA in the floor); OOM catch-retry loop (static P1 fixes prevent it; a pre-flight
estimate is the most warranted); the 33-frame expected-OOM baseline run on real 8GB.

## Acceptance (the FIT test, not a beauty contest)
On a memory-constrained config (or `--lowvram`): the hardened floor renders i2v at
832x480 x17, **measured peak < the engine's usable ceiling** (target ~8GB) at 0.1s
probe, euler/simple, GGUF UNET + GGUF/fp16 umt5 (no fp8), tiled decode -- and holds
the still. Then VERIFY-AT-BUILD on a real Mac (MPS) + AMD before claiming the
cross-platform floor. Promote defaults only after the fit passes; keep every knob
env-overridable so bigger cards tune up.

## Verify-at-build (from /object_info + real backends)
`CLIPLoaderGGUF` inputs + umt5-GGUF support; `VAEDecodeTiled` required inputs + IMAGE
output index 0; ComfyUI-GGUF dequant on MPS/ROCm/DirectX; real 8GB peak.
