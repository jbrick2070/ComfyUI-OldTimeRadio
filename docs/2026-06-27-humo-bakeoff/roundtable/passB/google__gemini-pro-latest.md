<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. A fully-resident 17B GGUF mathematically busts the 14.5 GB ceiling, and the "no eng_humo edit" constraint blocks the required memory-management fixes.

1. **Ranked Rec:** Keep-1.7B-harden. 
   *Why:* The 17B Q5 already scored a mediocre 30/45, Q3 will be worse, and fitting an 11.86 GB UNET plus umt5/whisper/VAE into 14.5 GB requires `free_after_use` eviction, which `eng_humo` explicitly forbids.

MUST-FIX BEFORE BUILD:
1. **[DECISION to harden] Fatal VRAM Math / Fully Resident Assumption:** The plan assumes a 17B Q5_K_M (11.86 GB) can fit the <=14.5 GB ceiling. However, `eng_humo.py` lines 259-262 explicitly enforce "Render FULLY RESIDENT... No free_after_use". 11.86 GB (UNET) + ~4.7 GB (umt5) + ~1.5 GB (whisper) = ~18 GB, guaranteeing an OOM. 
   *Fix:* The bakeoff script (`run_humo_bakeoff.py`) must bypass `eng_humo.render_clip` entirely and call `wrapper_bridge.run_graph` directly with `free_after_use=True` (mirroring the `eng_wan_i2v` pattern at line 269).
2. **[Code facts] GGUF Node Resolution Blocker:** The plan notes `eng_humo._node_candidates` lacks `UnetLoaderGGUF` but mandates "no eng_humo edit". If the bakeoff relies on `eng_humo.load()`, it will fail-closed because the GGUF class won't resolve. 
   *Fix:* The bakeoff script must manually import and resolve `UnetLoaderGGUF`, merging it into the `classes` dictionary passed to `run_graph`.
3. **[DECISION to harden] LoRA Bypass Wiring:** The 17B must run LoRA-free due to shape mismatch. `eng_humo._build_graph` (lines 203-206) only skips the LoRA if the name is explicitly "none", "skip", or "off". 
   *Fix:* The bakeoff builder must explicitly inject `"lora_name": "none"` into the graph payload and ensure the `modelsampling` node's `model` input is wired directly to `unet` (bypassing the `lora` node).

SHOULD-FIX:
1. **[DECISION to harden] Q3 vs Q5 priority:** If you proceed with the 17B GGUF leg, test the 8.4 GB Q3_K_M first. It is the only variant with a mathematical prayer of fitting without aggressive inter-node eviction. If Q3's quality is visibly worse than the 1.7B tier, you can abort the Q5 VRAM fight entirely.

[ASSUMPTION] The plan assumes the WanHuMoImageToVideo audio cross-attention mechanism will accept a GGUF-loaded UNET without throwing a tensor type/device mismatch error. This is a high-risk assumption for ComfyUI custom nodes that expect standard fp8/fp16 safetensors. The 33f smoke test must be run *first* before any quality or VRAM profiling.