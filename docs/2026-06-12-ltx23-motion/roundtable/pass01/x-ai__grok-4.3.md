<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The plan's 22B+gemma+i2v path directly contradicts the only implemented graph+loader in eng_ltx_video.py (which hard-wires 2B v0.9 + t5xxl + CLIPLoader and never instantiates a 22B transformer or gemma node).

MUST-FIX BEFORE BUILD:
1. [Questions for the panel / Q1 + eng_ltx_video.py:115] The 23.5 GB fp8 transformer + 8.8 GB gemma cannot fit under 14.5 GB even sequentially; the only VRAM mitigation present is free_after_use on the t5 path. Add explicit encode-then-free gemma + block-swap offload before any 22B load, or the assert_vram_within_ceiling call will fail at runtime.
2. [The reference that DOES move + eng_ltx_video.py:280] _node_candidates and _node_candidates_sampling only ever return CLIPLoader (type=ltxv) and t5xxl_fp16; LTXVGemmaCLIPModelLoader is mentioned only in a comment as "operator-gated" and never wired. Either implement the gemma graph branch or remove all 22B+gemma references.
3. [On disk already + eng_ltx_video.py:200] _ckpt_path only probes for ltx-video-2b-v0.9.safetensors; 22B files are never returned. Change the candidate list and _ckpt_name logic to accept the 23.5 GB fp8 file when OTR_LTX_VIDEO_CKPT points to it.
4. [eng_ltx_video.py:320] _build_graph_i2v and _node_candidates_i2v assume LTXVImgToVideoConditionOnly exists and accepts (vae, image, latent, strength); the motion-sweep md shows the identical node but no fallback or version check. Add an explicit presence assertion in assert_usable or the resolve_graph_classes call will raise at first i2v render.
5. [What we run TODAY + 2026-06-12-ltx-motion-sweep.md] The sweep that produced MAD 4.21 used 257 frames at 768x448; the decode-floor logic in _ltx_frame_length and eng_ltx_video.py:95 still defaults to 169 and only warns at 1472x832. Raise the documented safe length for i2v or the VAEDecode tensor mismatch will recur.

SHOULD-FIX:
1. [eng_ltx_video.py:240] _use_distilled_lora only checks for "22b" in the filename and the distilled LoRA; the camera-control dolly LoRA from the Goofer reference is absent from both disk inventory and code. Either add OTR_LTX_CAMERA_LORA wiring or document that the motion driver will be missing.
2. [Questions for the panel / Q2] No evidence that the base 22B distilled model supplies motion without the LoRA; the only motion delta proven in the md is euler_cfg_pp + length. Add an explicit ablation note before claiming the LoRA is highest-leverage.
3. [eng_ltx_video.py:410] render_clip calls assert_vram_within_ceiling only when OTR_TEST_MODE is unset; the 16 GB ceiling must be enforced on every path including distilled+LoRA.

OPTIONAL / NICE-TO-HAVE:
- Expose OTR_LTX_LENGTH as an env var instead of only the decode floor so the 257-frame motion recipe can be selected without editing _LTX_MAX_FRAMES_DEFAULT.
- Log the actual VRAM peak (via NVML) next to the MAD score in the smoke harness.

CUT THESE (over-engineering):
1. LtxOrbitEngine class and all _ORBIT_PROMPT_DEFAULT wiring: the motion-sweep md already achieves REAL motion via sampler+length alone; the prompt preset adds no new node or LoRA and can be done in the caller's text_prompt.
2. The entire _SigmasFromValues inner class and ManualSigmas indirection: LTX_DISTILLED_SIGMAS is only eight floats; inlining them into the sampleradv node inputs removes one non-standard class with no behavior change.

[ASSUMPTION] The plan assumes a ComfyUI LTX 2.3 wrapper that can load the 23.5 GB fp8 transformer and gemma encoder; no such loader appears in the provided grounding excerpts. [ASSUMPTION] DynamicVRAM + sequential offload will keep peak <=14.5 GB for any 22B config; the only measured ceiling in the code is the t5xxl case.