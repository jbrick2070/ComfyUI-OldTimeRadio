# wan_ti2v floor roundtable -- pass02 judgment (CONVERGED)

Panel: GPT-5.5, Opus-4.8, Gemini-3.1-pro, Grok-4.3, DeepSeek-v4-pro, Sonnet-4.6.
Spend ~$1.04 (pass01+02 total ~$1.94).

## Convergence call: CONVERGED -- stop at pass02
pass02 produced **no new strategic material**. Opus + Gemini flipped to
"yes-with-fixes"; the other four "no" verdicts are ALL "the plan describes the right
fix but the engine code doesn't implement it yet -- here are the exact values."
That is an implementation checklist, not a strategy disagreement. The pass01
GGUF-vs-fp8 conflict is fully resolved (fp8 Mac-broken; GGUF is the path). A pass03
would only re-confirm; per cost discipline we stop. (Operator OK'd 2-3; 2 converged.)

## Unanimous build checklist (fold into the final plan)
1. **CLIP off fp8 (the Mac fix):** add `_clip_loader_mode()` mirroring `_loader_mode()`;
   floor default = GGUF umt5 (`umt5-xxl-encoder-Q5_K_M.gguf`) via `CLIPLoaderGGUF`
   (env `OTR_WAN_TI2V_CLIP_LOADER`, gguf|safetensors); fp16 umt5 = the safetensors
   fallback. Make `_node_candidates["clip"]` conditional + fix the clip-node inputs.
2. **Tiled VAE:** conditional candidate gated on `OTR_WAN_TI2V_TILED_VAE` (default 1)
   -> `VAEDecodeTiled` vs `VAEDecode`; add `tile_size` (~256 for 8GB) + `overlap` to
   the `vaedecode` inputs dict; floor fails-closed if the node is absent.
3. **Frames (min-drop ALONE is insufficient -- unanimous):** `_TI2V_MIN_FRAMES` 33->17,
   add `_TI2V_DEFAULT_FRAMES=17`, AND fix the `render_clip` fallback
   `plan["target_frame_count"] or self.target_fps` -> `or _TI2V_DEFAULT_FRAMES`
   (today an omitted count = 25fps misused as frames -> quantizes to 33 -> OOMs).
   Clamp floor-tier requests so upstream can't ask for 33+.
4. **Sampler:** default `uni_pc`->`euler` in `_build_graph` AND read the SAME default
   in `assert_usable` (so unset passes the whitelist); `_PORTABLE_SAMPLERS={"euler"}`,
   fail-closed on non-whitelisted (Opus caught the self-reject trap).
5. **Probe:** `VramPeakProbe(interval_s=1.0)` -> `0.1` for the measurement.
6. **Shared config resolver** for steps/cfg/shift/sampler/scheduler -- range-checked,
   used by BOTH `assert_usable` and `_build_graph` (raw int()/float() crashes today).
7. **VAE guard:** whitelist the approved Wan2.2 VAE name (today only empty/2.1 reject).
8. **Offload:** `--lowvram`/sequential is a ComfyUI SERVER-LAUNCH flag; the engine is
   in-process and cannot set it per-render -> operator DOC, not engine control.

## Cut (unanimous -- dead code / out of scope)
- CFG/steps distill guard + LoRA license guard (no LoRA in the floor graph).
- OOM catch-and-retry loop (P1 static fixes prevent OOM; a pre-flight estimate is
  the most that's warranted).
- The mandatory 33-frame "expected-OOM" baseline run on real 8GB (known to OOM;
  run only on bigger HW as a negative control).

## VERIFY-AT-BUILD (from /object_info + real backends)
- `CLIPLoaderGGUF` input schema + umt5-GGUF support; `VAEDecodeTiled` required inputs
  + IMAGE output index 0 (so `results[_TERMINAL][0]` holds).
- ComfyUI-GGUF dequant on real MPS / ROCm / DirectX (pure-torch, expected yes).
- Real 8GB peak measurement at 0.1s probe; confirm 832x480 actually resolves.
