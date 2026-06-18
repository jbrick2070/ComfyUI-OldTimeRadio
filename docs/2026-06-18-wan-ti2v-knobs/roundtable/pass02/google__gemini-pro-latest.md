<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The VRAM and portability strategies correctly address the 8GB/Mac constraints, but the graph spec and node candidates are missing required keys to execute the new nodes.

MUST-FIX BEFORE BUILD:
1. [P1 - VRAM] `VAEDecodeTiled` graph inputs. The plan adds `VAEDecodeTiled` to the `vaedecode` node candidates, but `_build_graph` only supplies `samples` and `vae`. ComfyUI's `VAEDecodeTiled` requires a `tile_size` input. Fix: Add `"tile_size": 512` (or appropriate default) to the `"vaedecode"` inputs dict in `_build_graph`.
2. [P2 - Portability] Missing GGUF CLIP candidate. The plan dictates moving the umt5 CLIP to a GGUF file, but `_node_candidates()` hardcodes `"clip": ("CLIPLoader",)`. The standard `CLIPLoader` will fail on a `.gguf` file. Fix: Add `"CLIPLoaderGGUF"` to the `"clip"` tuple in `_node_candidates()`.
3. [P2 - Portability] Hardcoded sampler fallback. The plan changes the default sampler to `euler`, but `_build_graph` currently has `sampler = os.environ.get("OTR_WAN_TI2V_SAMPLER", "uni_pc")`. Fix: Change the fallback value in `_build_graph` to `"euler"`.

SHOULD-FIX:
4. [P1 - VRAM] Peak probe interval. The plan calls for measuring at `interval_s=0.1`, but `render_clip` currently hardcodes `_MC.VramPeakProbe(interval_s=1.0)`. Fix: Update the instantiation in `render_clip` to `interval_s=0.1` to catch transient decode spikes.

CUT THESE (over-engineering):
1. [P3 - Reliability] "CFG/steps coupling guard... if a distill is ever configured". Safe to cut: The plan explicitly states Lightning/distill LoRAs are CUT and out of scope for this floor. Do not write dead code for hypothetical features.
2. [P3 - Reliability] "License guard... IF a LoRA path is ever added". Safe to cut: Same reason. LoRAs are out of scope.
3. [P3 - Reliability] "OOM: a pre-flight frame/size estimate or a catch+retry-at-lower-length". Safe to cut: The entire point of P1 (17 frames, Tiled VAE, strict 8GB measurement) is to statically guarantee the floor fits. Catch+retry adds massive state/graph complexity to handle an OOM that P1 is designed to prevent.