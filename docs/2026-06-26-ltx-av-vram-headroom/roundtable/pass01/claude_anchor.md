# Claude anchor review -- R2 (coding plan / implementability)

Grounded against: eng_ltx_av.py, wrapper_bridge.py, render_driver.py, the live
render log, nvidia-smi, the on-disk quant inventory.

## VERDICT

Diagnosis CONFIRMED. The primary fix DIRECTION (force the AV unet to load with
activation headroom instead of to the brim) is correct, but it has ONE
load-bearing unverified assumption that must be settled before coding, plus an
exception-safety requirement. Not yet build-ready; 3 MUST-FIX.

## MUST-FIX

1. [BIGGEST RISK -- UNVERIFIED] Does ComfyUI's GGUF unet loader even honor
   PARTIAL/lowvram offload for this quantized unet? The whole fix assumes we can
   make the 10537 MB unet load partially (like `ltx_video` did at 5455+5081).
   Some GGUF paths pin the full dequantized tensor set and ignore the
   free-memory budget. If the GGUF unet cannot be partially offloaded, reserving
   VRAM just causes an OOM/abort, not a graceful partial load -- fix A is inert
   and we fall to B/C. GROUND THIS FIRST (read the GGUF loader's lowvram
   handling) before committing to A.

2. [CONFIRMED] The reserve mechanism must be EXCEPTION-SAFE. `render_clip` raises
   `GraphExecutionError` on the no-fallback path; a global reserved-VRAM bump
   that is not restored in a `finally` leaks into the next engine's load and
   silently degrades it. Prefer passing a per-call `memory_required` hint to the
   loader over mutating a module global; if a global is the only lever, wrap
   `run_graph` in try/finally.

3. [UNVERIFIED -- locate the lever] comfy/model_management.py was NOT found under
   C:\Users\jeffr\Documents\ComfyUI (Desktop install keeps core elsewhere). The
   exact reserved-VRAM symbol (global vs `load_models_gpu(minimum_memory_required=)`
   param vs `extra_reserved()`) is therefore UNCONFIRMED. The plan names it only
   as "e.g." -- pin the real symbol on THIS build before writing the hook.

## SHOULD-FIX

- The b002=25 / b003=6.84 / b004=223 NON-monotonicity at near-identical usable
  VRAM points at allocator state ACROSS beats (caching/fragmentation), not only
  per-beat headroom. Even with headroom, B (PYTORCH_CUDA_ALLOC_CONF) likely
  matters. But VERIFY torch 2.10 Windows support of `expandable_segments`
  empirically with a tiny offline probe (NOT during a live render) -- do not ship
  an env knob that no-ops on this platform.

- Pin the headroom EMPIRICALLY, not as a 2-3 GB guess: disable the NVIDIA sysmem
  fallback (loud OOM), then lower the reserve until the AV beat just fits -> that
  is the real audio-activation peak. Hard-code that + a margin.

## UNVERIFIABLE (verify-at-build)

- Whether `AudioVAE` (693 MB) stays resident through the denoise loop. V-1 says
  the audio-latent branch is never wired (the clip is always silent), so AudioVAE
  may not load for the video path at all -- but `keep={"unet",TERMINAL,"lora"}`
  does not mention it. Trace the graph; if it co-resides, evict it pre-denoise.

## Invariants I will not let a panel "fix" break

No fallbacks / fail LOUD; never `unload_all_models`; change WIRED + ON in the
canonical workflow JSON same-change; <= 14.5 GB ceiling; UTF-8 no BOM; push to
v2.0-alpha.
