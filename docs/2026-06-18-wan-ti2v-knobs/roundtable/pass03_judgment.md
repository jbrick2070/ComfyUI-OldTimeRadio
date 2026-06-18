# wan_ti2v floor roundtable -- pass03 judgment (CONVERGED, loop stopped)

Panel: Opus-4.8, Sonnet-4.6, GPT-5.5, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro.
Spend ~$1.21 (pass01+02+03 total ~$3.15).

## Convergence: CONFIRMED -- stop at pass03 (no pass04)
Verdicts: 3 yes-with-fixes (Opus/Sonnet/Gemini), 3 no (GPT/Grok/DeepSeek). The "no"s
raise ZERO new strategy -- they are hyper-granular implementation nits on the SAME
checklist (exact tile_size, exact VAE-name whitelist values, raw-env parsing already
covered by the shared resolver) plus re-flagging the node-schema checks that are
inherently BUILD-TIME verifies (/object_info), not resolvable from code by any panel.
The method (pass02_plan.md) stands. Further passes would only re-confirm.

## Folded into the final plan (the genuinely useful pass03 deltas)
1. **Frame-clamp bypass (Gemini/Opus/DeepSeek -- the one real catch):** dropping
   `_TI2V_MIN_FRAMES`/adding `_TI2V_DEFAULT_FRAMES` is NOT enough -- `render_clip`
   honors `plan["target_frame_count"]` BEFORE the fallback (capped only at 177), so an
   upstream 33 still reaches the OOM path. The floor must CLAMP the effective length:
   `length = min(requested or _TI2V_DEFAULT_FRAMES, _TI2V_FLOOR_MAX)` unless an
   explicit higher-tier override env (`OTR_WAN_TI2V_MAX_FRAMES`) is set AND host caps
   allow it. This is the load-bearing reliability item.
2. **Name the constants concretely:** `_TI2V_DEFAULT_FRAMES = 17`,
   `_WAN22_VAE_ALLOWED = frozenset({"wan2.2_vae.safetensors"})`, new floor CLIP
   default constant (e.g. `_TI2V_DEFAULT_CLIP_GGUF = "umt5-xxl-encoder-Q5_K_M.gguf"`)
   -- don't leave them as prose.
3. **Single resolver does ALL env parsing** (steps/cfg/shift/sampler/scheduler/frames/
   tile) with range-checks, shared by assert_usable + _build_graph -- no raw int()/
   float() anywhere in _build_graph.

## Re-confirmed VERIFY-AT-BUILD (not resolvable without /object_info + real HW)
- `VAEDecodeTiled` required inputs (tile_size/overlap defaulted by server? IMAGE out
  idx 0) and `CLIPLoaderGGUF` input schema (does it take type="wan"/device?).
- ComfyUI-GGUF dequant on real MPS / ROCm / DirectX; real 8GB peak at 0.1s probe.
These are BUILD-TIME checks against a live server, by design -- not a plan defect.

## FINAL = pass02_plan.md + the 3 deltas above. Build-ready.
